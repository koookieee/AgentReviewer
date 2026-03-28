"""
Paper Reviewer Generator — subclasses HarborGenerator for review tasks without a verifier.

When verifier.disable=true in Harbor config, results.verifier_result is None.
The base HarborGenerator treats this as a failure and retries. This subclass
instead computes rewards from the review output itself.
"""

import asyncio
import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from loguru import logger
from uuid import uuid4

from skyrl.train.generators.base import TrajectoryID
from skyrl.train.generators.utils import get_response_ids_and_loss_mask_from_messages
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig

from .harbor_generator import HarborGenerator, HarborAgentOutput, MAX_NUM_RETRIES_PER_TRIAL


# Directories to skip when uploading task content to E2B sandbox
_TASK_SKIP_DIRS = {"environment", ".git", "__pycache__"}


class PaperReviewerTrial(Trial):
    """Trial subclass that uploads task content files to E2B sandboxes.

    Harbor's E2B environment doesn't include task content (latex/, etc.) in the
    sandbox because `Template.from_dockerfile` has no build context. Docker
    environments get them via bind-mount. This subclass uploads the task content
    directory after environment setup so the agent can actually read the paper.
    """

    async def _setup_environment(self) -> None:
        await super()._setup_environment()
        # Upload task content (everything except environment/) to the sandbox workdir
        task_dir = Path(self.config.task.path)
        if not task_dir.is_dir():
            logger.warning(f"Task dir {task_dir} not found, skipping file upload")
            return
        workdir = self._environment._workdir or "/"
        for item in sorted(task_dir.iterdir()):
            if item.name in _TASK_SKIP_DIRS:
                continue
            target = f"/{workdir.strip('/')}/{item.name}"
            try:
                if item.is_file():
                    await self._environment.upload_file(item, target)
                elif item.is_dir():
                    await self._environment.upload_dir(item, target)
            except Exception as e:
                logger.warning(f"Failed to upload {item.name} to {target}: {e}")
        logger.info(f"Uploaded task content from {task_dir} to {workdir}")

        # Upload search-papers-skill.md from the harbor examples dir
        skill_file = Path(__file__).parent / "search-papers" / "SKILL.md"
        if skill_file.is_file():
            target = f"/{workdir.strip('/')}/search-papers-skill.md"
            try:
                await self._environment.upload_file(skill_file, target)
                logger.info(f"Uploaded search skill to {target}")
            except Exception as e:
                logger.warning(f"Failed to upload search skill: {e}")

        # Write search API URL so the agent can discover it
        search_api_url = os.environ.get("SEARCH_API_URL", "")
        if search_api_url:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(search_api_url)
                tmp_path = f.name
            target = f"/{workdir.strip('/')}/search_api_url.txt"
            try:
                await self._environment.upload_file(Path(tmp_path), target)
                logger.info(f"Uploaded search API URL ({search_api_url}) to {target}")
            except Exception as e:
                logger.warning(f"Failed to upload search API URL: {e}")
            finally:
                os.unlink(tmp_path)


# Required sections in the review output (from idea-reviewer.md output format)
REQUIRED_SECTIONS = [
    "Novelty Assessment",
    "Impact Analysis",
    "Literature Gaps",
    "Methodological Concerns",
    "Positioning Recommendations",
    "Overall Verdict",
]

# Required numeric scores in the Overall Verdict section
REQUIRED_SCORES = [
    "Novelty",
    "Impact",
    "Rigor",
    "Positioning",
    "Overall",
]


def compute_format_reward(chat_history: list[dict], task_path: str = "") -> float:
    """Compute a reward based on how well the review follows the expected format.

    Checks for:
    - Presence of all 6 required sections (60% of score)
    - Presence of all 5 numeric scores X/10 (25% of score)
    - Non-trivial review length (15% of score)

    Returns:
        float in [0.0, 1.0]
    """
    # Extract the last assistant message — this should contain the final review
    last_assistant_content = ""
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            last_assistant_content = msg.get("content", "")
            break

    if not last_assistant_content:
        return 0.0

    reward = 0.0

    # --- Section presence (60% weight) ---
    section_weight = 0.6 / len(REQUIRED_SECTIONS)
    for section in REQUIRED_SECTIONS:
        # Check for section header (case-insensitive, with or without ### prefix)
        pattern = rf"(?:^|\n)\s*#{{0,4}}\s*{re.escape(section)}"
        if re.search(pattern, last_assistant_content, re.IGNORECASE):
            reward += section_weight

    # --- Numeric scores (25% weight) ---
    score_weight = 0.25 / len(REQUIRED_SCORES)
    for score_name in REQUIRED_SCORES:
        # Match patterns like "**Novelty**: 7/10" or "Novelty: 8/10"
        pattern = rf"\*{{0,2}}{re.escape(score_name)}\*{{0,2}}\s*:\s*\d+\s*/\s*10"
        if re.search(pattern, last_assistant_content, re.IGNORECASE):
            reward += score_weight

    # --- Review length (15% weight) ---
    word_count = len(last_assistant_content.split())
    if word_count >= 500:
        reward += 0.15
    elif word_count >= 200:
        reward += 0.15 * (word_count - 100) / 400  # Linear ramp from 100 to 500 words

    return min(reward, 1.0)


def compute_dummy_reward(chat_history: list[dict], task_path: str = "") -> float:
    """Always returns 1.0 — for pipeline testing."""
    return 1.0


REWARD_FUNCTIONS = {
    "dummy": compute_dummy_reward,
    "format": compute_format_reward,
}


def _extract_chat_history_from_trial(trial: Trial) -> tuple[list[dict] | None, int, int]:
    """Extract chat history from Claude Code's session JSONL when metadata is unavailable.

    Harbor's Claude Code agent writes session JSONL files but does not populate
    context.metadata["all_messages"] (unlike terminus-2). This function reads
    the JSONL directly and reconstructs a chat history compatible with the
    training pipeline.

    Returns:
        (chat_history, num_turns, summarization_count) or (None, 0, 0) on failure.
    """
    agent_dir = trial._trial_paths.agent_dir
    projects_dir = agent_dir / "sessions" / "projects"
    if not projects_dir.is_dir():
        return None, 0, 0

    # Find JSONL files across all project dirs
    jsonl_files = list(projects_dir.rglob("*.jsonl"))
    if not jsonl_files:
        return None, 0, 0

    # Use the largest JSONL (most content)
    jsonl_path = max(jsonl_files, key=lambda p: p.stat().st_size)

    messages: list[dict] = []
    num_turns = 0
    try:
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            event_type = event.get("type")

            if event_type == "user":
                msg = event.get("message", {})
                content = msg.get("content", "")
                if content:
                    messages.append({"role": "user", "content": content})

            elif event_type == "assistant":
                msg = event.get("message", {})
                content_blocks = msg.get("content", [])
                # Reconstruct assistant message text
                text_parts = []
                for block in content_blocks:
                    if isinstance(block, dict):
                        if block.get("type") == "text" and block.get("text"):
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            text_parts.append(
                                f"[Tool use: {block.get('name', '')}({json.dumps(block.get('input', {}))})]"
                            )
                combined = "\n".join(text_parts)
                if combined:
                    # Wrap content with think tags if missing so the chat template
                    # tokenizes correctly (avoids generation prompt assertion error).
                    if "<think>" not in combined:
                        combined = f"<think>\n...\n</think>\n\n{combined}"
                    messages.append({"role": "assistant", "content": combined})
                    num_turns += 1

            elif event_type == "tool_result":
                # Tool results appear as user messages in the training format
                content = event.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        b.get("text", "") for b in content if isinstance(b, dict)
                    )
                if content:
                    messages.append({"role": "user", "content": str(content)})

    except Exception as e:
        logger.warning(f"Failed to parse session JSONL {jsonl_path}: {e}")
        return None, 0, 0

    if len(messages) <= 1:
        return None, 0, 0

    return messages, num_turns, 0


class PaperReviewerGenerator(HarborGenerator):
    """HarborGenerator subclass that computes rewards from review output when no verifier is available."""

    def __init__(
        self,
        generator_cfg,
        harbor_cfg,
        inference_engine_client,
        tokenizer,
        max_seq_len: int,
        reward_type: str = "format",
    ):
        super().__init__(
            generator_cfg=generator_cfg,
            harbor_cfg=harbor_cfg,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        self.reward_type = reward_type

        if reward_type == "llm_judge":
            from .Reward.llm_judge import LLMJudgeReward

            self._reward_fn = LLMJudgeReward()
            self._reward_is_async = True
        elif reward_type in REWARD_FUNCTIONS:
            self._reward_fn = REWARD_FUNCTIONS[reward_type]
            self._reward_is_async = False
        else:
            valid = list(REWARD_FUNCTIONS.keys()) + ["llm_judge"]
            raise ValueError(f"Unknown reward_type '{reward_type}'. Choose from: {valid}")

        logger.info(f"PaperReviewerGenerator initialized with reward_type='{reward_type}'")

    async def harbor_agent_loop(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
    ) -> HarborAgentOutput:
        """Run a single Harbor agent trial, computing reward from review output when verifier is disabled.

        This overrides the parent method to handle the case where verifier_result is None
        (i.e., verifier.disable=true). Instead of retrying on missing verifier results,
        we extract the chat history and compute reward from the review content.
        """
        reward = None
        chat_history = None
        summarization_count = None
        num_turns = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False

        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i+1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            try:
                # Create a fresh Trial each attempt so agent state is clean on retry.
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                config["agent"]["kwargs"]["session_id"] = uuid4().hex
                trial_config = TrialConfig.model_validate(config)
                trial = PaperReviewerTrial(trial_config)

                async with self._rate_limiter:
                    results = await trial.run()

                # Parse exception type
                exc_type = results.exception_info.exception_type if results.exception_info else None
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = exc_type == "AgentTimeoutError"

                # --- Handle fatal errors first ---
                if is_agent_timeout_error:
                    logger.debug(f"{prefix} hit AgentTimeoutError (no retry). Results: {results}")
                    break

                if is_context_length_error:
                    logger.debug(
                        f"{prefix} hit ContextLengthExceededError, will train with reward=0. Results: {results}"
                    )
                    reward = 0

                # --- Extract chat history (before reward determination) ---
                if results.agent_result and results.agent_result.metadata:
                    # terminus-2 style: metadata populated directly
                    chat_history = results.agent_result.metadata.get("all_messages")
                    summarization_count = results.agent_result.metadata.get("summarization_count", 0)
                    num_turns = results.agent_result.metadata.get("n_episodes", 0)
                else:
                    # Claude Code style: read from session JSONL
                    chat_history, num_turns, summarization_count = _extract_chat_history_from_trial(trial)

                if not chat_history or len(chat_history) <= 1 or chat_history[0]["role"] != "user":
                    logger.warning(
                        f"{prefix} failed: Invalid chat history. "
                        f"chat_history: {chat_history}\nResults: {results}"
                    )
                    continue

                # --- Determine reward ---
                if reward is None:
                    # reward is already set to 0 for context_length_error above
                    if results.verifier_result:
                        # Verifier ran (shouldn't happen with disable=true, but handle gracefully)
                        reward = results.verifier_result.rewards["reward"]
                    else:
                        # No verifier — compute reward from the review output
                        if self._reward_is_async:
                            reward = await self._reward_fn(chat_history, prompt)
                        else:
                            reward = self._reward_fn(chat_history, prompt)
                        logger.debug(
                            f"{prefix} computed {self.reward_type} reward={reward:.3f} "
                            f"(no verifier result)"
                        )

                successful = True
                logger.debug(f"{prefix} successful: reward={reward}. Results: {results}")
                break

            except Exception as e:
                logger.warning(f"{prefix} failed: Error running trial: {e}. Results: {results}")
                continue

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            error_message = (
                f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}), "
                f"will set loss mask to [0]."
            )
            if stop_reason == "error":
                error_message += f" Results: {results}"
            logger.warning(error_message)
            return HarborAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason=stop_reason,
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
            )

        # --- Tokenize prompt and response ---
        assert chat_history[0]["role"] == "user", "The first message should be a user message"
        prompt_msgs = [chat_history[0]]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_msgs,
            add_generation_prompt=False,
            return_dict=False,
            tokenize=True,
            chat_template=self.custom_chat_template_content,
        )
        initial_prompt_length = len(prompt_ids)

        # Process response messages (everything after the first message)
        response_messages = chat_history[1:]
        assistant_logprobs = getattr(results.agent_result, "output_logprobs", None)
        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            response_messages, self.tokenizer, assistant_logprobs, chat_template=self.custom_chat_template_content
        )

        # Determine stop reason
        max_response_tokens = max(0, self.max_seq_len - initial_prompt_length)
        if is_context_length_error or len(response_ids) > max_response_tokens:
            stop_reason = "context_length"
        else:
            stop_reason = "complete"

        # Apply overlong filtering
        if self.generator_cfg.apply_overlong_filtering and stop_reason == "context_length":
            loss_mask = [0] * len(loss_mask)

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return HarborAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            trajectory_id=trajectory_id,
            summarization_count=summarization_count,
            num_turns=num_turns,
        )
