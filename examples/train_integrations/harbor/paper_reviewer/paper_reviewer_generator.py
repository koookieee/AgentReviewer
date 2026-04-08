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

from ..harbor_generator import HarborGenerator, HarborAgentOutput, MAX_NUM_RETRIES_PER_TRIAL


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

        # Upload search skill into ~/.claude/skills/search-papers/ so Claude Code
        # discovers it natively. Harbor's setup command copies ~/.claude/skills/
        # into CLAUDE_CONFIG_DIR/skills/ before the agent starts.
        skill_file = Path(__file__).parent / "search" / "SKILL.md"
        if skill_file.is_file():
            skill_target = "/root/.claude/skills/search-papers/SKILL.md"
            try:
                await self._environment.upload_file(skill_file, skill_target)
                logger.info(f"Uploaded search skill to {skill_target}")
            except Exception as e:
                logger.warning(f"Failed to upload search skill: {e}")
            # Also upload to workdir as fallback for direct file reading
            fallback_target = f"/{workdir.strip('/')}/search-papers-skill.md"
            try:
                await self._environment.upload_file(skill_file, fallback_target)
            except Exception:
                pass

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

        # Install tavily CLI so Claude Code can use web search
        tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
        if tavily_api_key:
            try:
                result = await self._environment.exec(
                    command=(
                        "pip install -q tavily-cli && "
                        f"tvly login --api-key {tavily_api_key}"
                    ),
                    timeout_sec=120,
                )
                logger.info(f"Installed tavily CLI in sandbox (exit={result.exit_code})")
            except Exception as e:
                logger.warning(f"Failed to install tavily CLI: {e}")


# ---- LLM Judge reward ----

# Load the judge prompt template once
_JUDGE_PROMPT_PATH = Path(__file__).parent / "llm_judge_instruction.md"
_JUDGE_PROMPT_TEMPLATE: str | None = None


def _get_judge_prompt_template() -> str:
    global _JUDGE_PROMPT_TEMPLATE
    if _JUDGE_PROMPT_TEMPLATE is None:
        _JUDGE_PROMPT_TEMPLATE = _JUDGE_PROMPT_PATH.read_text()
    return _JUDGE_PROMPT_TEMPLATE


def _extract_last_review(chat_history: list[dict]) -> str:
    """Extract the last assistant message (the final review)."""
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _load_task_metadata(task_dir: Path) -> dict:
    """Load title, abstract, and human reviews from a task directory.

    Looks for:
    - task_metadata.json: {"title": "...", "abstract": "...", "human_reviews": ["...", "..."]}
    - Or falls back to parsing latex/template.tex for title/abstract
    """
    metadata = {"title": "", "abstract": "", "human_reviews": []}

    # Try task_metadata.json first
    meta_path = task_dir / "task_metadata.json"
    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text())
            metadata["title"] = data.get("title", "")
            metadata["abstract"] = data.get("abstract", "")
            metadata["human_reviews"] = data.get("human_reviews", [])
            return metadata
        except Exception as e:
            logger.warning(f"Failed to parse {meta_path}: {e}")

    # Fallback: parse title/abstract from LaTeX
    tex_path = task_dir / "latex" / "template.tex"
    if tex_path.is_file():
        tex = tex_path.read_text(errors="replace")
        # Extract title
        m = re.search(r"\\title\{([^}]+)\}", tex)
        if m:
            metadata["title"] = m.group(1).strip()
        # Extract abstract
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
        if m:
            metadata["abstract"] = m.group(1).strip()

    # Load human reviews from human_reviews/ directory if present
    reviews_dir = task_dir / "human_reviews"
    if reviews_dir.is_dir():
        for review_file in sorted(reviews_dir.glob("*.md")) + sorted(reviews_dir.glob("*.txt")):
            metadata["human_reviews"].append(review_file.read_text(errors="replace"))

    # Or from a single human_reviews.json
    reviews_json = task_dir / "human_reviews.json"
    if reviews_json.is_file() and not metadata["human_reviews"]:
        try:
            metadata["human_reviews"] = json.loads(reviews_json.read_text())
        except Exception:
            pass

    return metadata


async def compute_llm_judge_reward(chat_history: list[dict], task_dir: Path) -> float:
    """Call an external LLM to evaluate the review quality.

    Uses the judge prompt from llm_judge_instruction.md, filled with:
    - Paper title and abstract
    - Human reference reviews
    - The model's generated review

    Returns a composite reward in [0.0, 1.0].
    """
    from openai import AsyncOpenAI

    model_review = _extract_last_review(chat_history)
    if not model_review or len(model_review) < 100:
        logger.warning("LLM judge: review too short or empty, returning 0.0")
        return 0.0

    metadata = _load_task_metadata(task_dir)
    if not metadata["title"]:
        logger.warning(f"LLM judge: no title found in {task_dir}, returning 0.0")
        return 0.0

    # Format human reviews
    if metadata["human_reviews"]:
        human_reviews_text = "\n\n---\n\n".join(
            f"**Human Review {i+1}:**\n\n{review}"
            for i, review in enumerate(metadata["human_reviews"])
        )
    else:
        logger.warning(f"LLM judge: no human reviews in {task_dir}, skipping issue_overlap and calibration")
        human_reviews_text = "(No human reviews available for this paper.)"

    # Fill the judge prompt template
    prompt = _get_judge_prompt_template()
    prompt = prompt.replace("{title}", metadata["title"])
    prompt = prompt.replace("{abstract}", metadata["abstract"])
    prompt = prompt.replace("{human_reviews}", human_reviews_text)
    prompt = prompt.replace("{model_review}", model_review)

    # Call the judge LLM
    judge_model = os.environ.get("LLM_JUDGE_MODEL", "gemini-3-flash-preview")
    judge_api_key = os.environ.get("LLM_JUDGE_API_KEY", os.environ.get("GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    judge_base_url = os.environ.get("LLM_JUDGE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

    if not judge_api_key:
        logger.warning("LLM judge: no API key (set LLM_JUDGE_API_KEY or OPENAI_API_KEY), returning 0.0")
        return 0.0

    client = AsyncOpenAI(api_key=judge_api_key, base_url=judge_base_url)

    try:
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        judge_output = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"LLM judge API call failed: {e}, returning 0.0")
        return 0.0

    # Parse JSON from the judge output
    try:
        # Extract JSON block (may be wrapped in ```json ... ```)
        json_match = re.search(r"\{[\s\S]*\}", judge_output)
        if not json_match:
            logger.warning(f"LLM judge: no JSON found in output, returning 0.0")
            return 0.0
        scores = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"LLM judge: JSON parse failed: {e}, returning 0.0")
        return 0.0

    # Extract individual scores
    comprehension = float(scores.get("comprehension", {}).get("score", 0))
    substance = float(scores.get("substance_and_specificity", {}).get("score", 0))
    insight = float(scores.get("insight", {}).get("score", 0))
    issue_overlap = float(scores.get("issue_overlap", {}).get("score", 0))
    calibration = float(scores.get("calibration", {}).get("score", 0))

    # Composite reward — weighted sum
    if metadata["human_reviews"]:
        # Full reward with all 5 criteria
        reward = (
            0.20 * comprehension +
            0.25 * substance +
            0.25 * insight +
            0.20 * issue_overlap +
            0.10 * calibration
        )
    else:
        # No human reviews — only use the 3 standalone criteria
        reward = (
            0.30 * comprehension +
            0.35 * substance +
            0.35 * insight
        )

    logger.info(
        f"LLM judge scores: comprehension={comprehension}, substance={substance}, "
        f"insight={insight}, issue_overlap={issue_overlap}, calibration={calibration} "
        f"→ reward={reward:.3f}"
    )
    return reward


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


REWARD_TYPES = {"llm_judge"}


class PaperReviewerGenerator(HarborGenerator):
    """HarborGenerator subclass that computes rewards from review output when no verifier is available."""

    def __init__(
        self,
        generator_cfg,
        harbor_cfg,
        inference_engine_client,
        tokenizer,
        max_seq_len: int,
        reward_type: str = "llm_judge",
    ):
        super().__init__(
            generator_cfg=generator_cfg,
            harbor_cfg=harbor_cfg,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        if reward_type not in REWARD_TYPES:
            raise ValueError(f"Unknown reward_type '{reward_type}'. Choose from: {REWARD_TYPES}")
        self.reward_type = reward_type
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
                        reward = results.verifier_result.rewards["reward"]
                    else:
                        task_dir = Path(prompt)
                        reward = await compute_llm_judge_reward(chat_history, task_dir)
                        logger.info(
                            f"{prefix} LLM judge reward={reward:.3f}"
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
