"""
Paper reviewer utilities for Tinker training — standalone extraction from
paper_reviewer_generator.py without skyrl dependencies.

Contains:
- PaperReviewerTrial: Trial subclass that uploads paper content to E2B
- compute_llm_judge_reward: Call Gemini to evaluate review quality
- _extract_chat_history_from_trial: Read Claude Code session JSONL
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import logging
from pathlib import Path

from harbor.trial.trial import Trial

logger = logging.getLogger(__name__)

# Directories to skip when uploading task content to E2B sandbox
_TASK_SKIP_DIRS = {"environment", ".git", "__pycache__"}

# Path to search skill and judge prompt — supports both repo layout
# (harbor/tinker/ → harbor/paper_reviewer/) and flat remote layout
# (/root/tinker_training/ → /root/tinker_training/paper_reviewer/)
_PAPER_REVIEWER_DIR = (
    Path(__file__).resolve().parent.parent / "paper_reviewer"
    if (Path(__file__).resolve().parent.parent / "paper_reviewer").is_dir()
    else Path(__file__).resolve().parent / "paper_reviewer"
)


def _trim_to_conclusion(content: str) -> str:
    """Trim paper content to end of Conclusion section, dropping appendices/references.

    Handles both Markdown (# Conclusion) and LaTeX (\\section{Conclusion}) formats.
    Returns original content unchanged if no Conclusion section is found.
    """
    m = re.search(
        r"^(# Conclusion|\\section\*?\{Conclusion\})",
        content,
        re.MULTILINE | re.IGNORECASE,
    )
    if not m:
        return content

    # Find the next top-level section after Conclusion to know where to cut
    after_conclusion = content[m.start():]
    next_section = re.search(r"^(# (?!Conclusion)|\\section\*?\{(?!Conclusion))", after_conclusion[1:], re.MULTILINE | re.IGNORECASE)
    end = m.start() + 1 + next_section.start() if next_section else len(content)

    trimmed = content[:end]
    logger.info(f"Trimmed paper to Conclusion: {len(content):,} → {end:,} chars")
    return trimmed


class PaperReviewerTrial(Trial):
    """Trial subclass that uploads task content files to E2B sandboxes.

    Harbor's E2B environment doesn't include task content (latex/, etc.) in the
    sandbox because Template.from_dockerfile has no build context. Docker
    environments get them via bind-mount. This subclass uploads the task content
    directory after environment setup so the agent can actually read the paper.
    """

    async def _upload_latex_trimmed(self, latex_dir: Path, target: str) -> None:
        """Upload latex/ dir, trimming template.tex to Conclusion before upload."""
        tex_file = latex_dir / "template.tex"
        if not tex_file.is_file():
            await self._environment.upload_dir(latex_dir, target)
            return

        original = tex_file.read_text(errors="replace")
        trimmed = _trim_to_conclusion(original)

        if trimmed == original:
            # No Conclusion found — upload as-is
            await self._environment.upload_dir(latex_dir, target)
            return

        # Write trimmed content to a temp file and upload in place of the original
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(trimmed)
            tmp_tex = Path(f.name)

        try:
            # Upload the rest of the latex dir first, then overwrite template.tex
            await self._environment.upload_dir(latex_dir, target)
            tex_target = f"{target.rstrip('/')}/template.tex"
            await self._environment.upload_file(tmp_tex, tex_target)
        finally:
            tmp_tex.unlink(missing_ok=True)

    async def _setup_environment(self) -> None:
        await super()._setup_environment()
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
                    if item.name == "latex":
                        await self._upload_latex_trimmed(item, target)
                    else:
                        await self._environment.upload_dir(item, target)
            except Exception as e:
                logger.warning(f"Failed to upload {item.name} to {target}: {e}")
        logger.info(f"Uploaded task content from {task_dir} to {workdir}")

        # Upload search skill into ~/.claude/skills/search-papers/ so Claude Code
        # discovers it natively. Harbor's setup command copies ~/.claude/skills/
        # into CLAUDE_CONFIG_DIR/skills/ before the agent starts.
        skill_file = _PAPER_REVIEWER_DIR / "search" / "SKILL.md"
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



# ── LLM Judge reward ────────────────────────────────────────────────────

_JUDGE_PROMPT_PATH = _PAPER_REVIEWER_DIR / "llm_judge_instruction.md"
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
    """Load title, abstract, and human reviews from a task directory."""
    metadata = {"title": "", "abstract": "", "human_reviews": []}

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

    # Fallback: parse title/abstract from template.tex (may be markdown or LaTeX)
    tex_path = task_dir / "latex" / "template.tex"
    if tex_path.is_file():
        tex = tex_path.read_text(errors="replace")
        # Markdown title: first # heading
        m = re.search(r"^# (.+)$", tex, re.MULTILINE)
        if m:
            metadata["title"] = m.group(1).strip()
        else:
            # LaTeX title fallback
            m = re.search(r"\\title\{([^}]+)\}", tex)
            if m:
                metadata["title"] = m.group(1).strip()
        # Markdown abstract: ## Abstract section
        m = re.search(r"^## Abstract\s*\n(.*?)(?=^#|\Z)", tex, re.MULTILINE | re.DOTALL)
        if m:
            metadata["abstract"] = m.group(1).strip()
        else:
            # LaTeX abstract fallback
            m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
            if m:
                metadata["abstract"] = m.group(1).strip()

    # Load human reviews
    reviews_dir = task_dir / "human_reviews"
    if reviews_dir.is_dir():
        for review_file in sorted(reviews_dir.glob("*.md")) + sorted(reviews_dir.glob("*.txt")):
            metadata["human_reviews"].append(review_file.read_text(errors="replace"))

    reviews_json = task_dir / "human_reviews.json"
    if reviews_json.is_file() and not metadata["human_reviews"]:
        try:
            metadata["human_reviews"] = json.loads(reviews_json.read_text())
        except Exception:
            pass

    return metadata


async def compute_llm_judge_reward(chat_history: list[dict], task_dir: Path) -> float:
    """Call an external LLM to evaluate the review quality.

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

    prompt = _get_judge_prompt_template()
    prompt = prompt.replace("{title}", metadata["title"])
    prompt = prompt.replace("{abstract}", metadata["abstract"])
    prompt = prompt.replace("{human_reviews}", human_reviews_text)
    prompt = prompt.replace("{model_review}", model_review)

    judge_model = os.environ.get("LLM_JUDGE_MODEL", "gemini-3-flash-preview")
    judge_api_key = os.environ.get("LLM_JUDGE_API_KEY", os.environ.get("GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    judge_base_url = os.environ.get("LLM_JUDGE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

    if not judge_api_key:
        logger.warning("LLM judge: no API key, returning 0.0")
        return 0.0

    client = AsyncOpenAI(api_key=judge_api_key, base_url=judge_base_url)

    try:
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16384,
        )
        judge_output = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"LLM judge API call failed: {e}, returning 0.0")
        return 0.0

    try:
        json_match = re.search(r"\{[\s\S]*\}", judge_output)
        if not json_match:
            logger.warning(f"LLM judge: no JSON found in output (len={len(judge_output)}, first 500 chars: {judge_output[:500]}), returning 0.0")
            return 0.0
        scores = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"LLM judge: JSON parse failed: {e}, returning 0.0")
        return 0.0

    comprehension = float(scores.get("comprehension", {}).get("score", 0))
    substance = float(scores.get("substance_and_specificity", {}).get("score", 0))
    insight = float(scores.get("insight", {}).get("score", 0))
    issue_overlap = float(scores.get("issue_overlap", {}).get("score", 0))
    calibration = float(scores.get("calibration", {}).get("score", 0))

    if metadata["human_reviews"]:
        reward = (
            0.20 * comprehension +
            0.25 * substance +
            0.25 * insight +
            0.20 * issue_overlap +
            0.10 * calibration
        )
    else:
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


# ── Chat history extraction ─────────────────────────────────────────────


def extract_chat_history_from_trial(trial: Trial) -> tuple[list[dict] | None, int, int]:
    """Extract chat history from Claude Code's session JSONL.

    Returns: (chat_history, num_turns, summarization_count) or (None, 0, 0) on failure.
    """
    agent_dir = trial._trial_paths.agent_dir
    projects_dir = agent_dir / "sessions" / "projects"
    if not projects_dir.is_dir():
        return None, 0, 0

    jsonl_files = list(projects_dir.rglob("*.jsonl"))
    if not jsonl_files:
        return None, 0, 0

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
                    if "<think>" not in combined:
                        combined = f"<think>\n...\n</think>\n\n{combined}"
                    messages.append({"role": "assistant", "content": combined})
                    num_turns += 1

            elif event_type == "tool_result":
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
