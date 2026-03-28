"""
LLM-as-Judge reward model using Gemini 3 Flash.

Evaluates AI-generated paper reviews on 7 rubrics that test genuine intellectual
engagement rather than surface-level quality signals. 6 rubrics are evaluated by
Gemini, 1 (citation count) is computed locally via regex.
"""

import asyncio
import os
import re
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# ---------------------------------------------------------------------------
# Pydantic models for Gemini structured output
# ---------------------------------------------------------------------------

class RubricScore(BaseModel):
    justification: str = Field(description="1-2 sentence rationale for the score")
    score: float = Field(description="Score from 0.0 to 1.0", ge=0.0, le=1.0)


class LLMJudgeOutput(BaseModel):
    paper_specificity: RubricScore
    intellectual_contribution: RubricScore
    evidence_claim_gap: RubricScore
    falsifiability: RubricScore
    literature_positioning: RubricScore
    technical_vs_surface: RubricScore


# ---------------------------------------------------------------------------
# Rubric weights (must sum to 1.0 with citation_count weight)
# ---------------------------------------------------------------------------

LLM_RUBRIC_WEIGHTS = {
    "paper_specificity": 0.10,
    "intellectual_contribution": 0.25,
    "evidence_claim_gap": 0.20,
    "falsifiability": 0.15,
    "literature_positioning": 0.10,
    "technical_vs_surface": 0.15,
}

CITATION_COUNT_WEIGHT = 0.05  # local rubric

# Sanity check
assert abs(sum(LLM_RUBRIC_WEIGHTS.values()) + CITATION_COUNT_WEIGHT - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
You are a meta-reviewer at a top ML conference (NeurIPS/ICML). You're evaluating \
whether an AI-generated review demonstrates genuine intellectual engagement with \
the paper, or is generic output that could apply to any paper.

## The Paper
<paper>
{paper_tex}
</paper>

## The Review
<review>
{review_text}
</review>

## Your Task

Evaluate this review on 6 dimensions. Be harsh — most AI reviews are mediocre. \
Reserve scores above 0.7 for genuinely impressive reviewing work.

1. **paper_specificity** (0.0-1.0): Does the review engage with specific technical \
details unique to THIS paper — architecture names, specific results, specific \
theorems — or could it apply to any paper in the subfield with minor edits?

2. **intellectual_contribution** (0.0-1.0): Does the review generate NEW understanding \
beyond what's in the paper? Non-obvious connections to other work, predicted failure \
modes, reframings of the contribution, identification of the real bottleneck. Does \
reading this review teach you something the paper didn't?

3. **evidence_claim_gap** (0.0-1.0): Does the review identify where the paper's CLAIMS \
(abstract, intro, conclusion) are not fully supported by the EVIDENCE (experiments, \
proofs, ablations)? Cross-references specific claims against specific results. \
Identifies statistical issues, missing controls, or broken theory-experiment links.

4. **falsifiability** (0.0-1.0): Does the review propose CONCRETE experiments or \
analyses that would change the reviewer's assessment? Not "add more experiments" \
but "if you ran X with Y controlled and saw Z, that would convince me." Tests \
whether the reviewer understands what evidence actually matters.

5. **literature_positioning** (0.0-1.0): Does the review correctly position this paper \
relative to existing work in a way that CHANGES the evaluation? Not just citing \
papers, but finding specific prior work that invalidates a novelty claim, a simpler \
baseline the authors missed, or a subfield that recontextualizes the contribution.

6. **technical_vs_surface** (0.0-1.0): What fraction of the review engages with actual \
science (methodology, algorithms, proofs, experimental design, statistical validity, \
complexity analysis) versus surface observations (writing quality, paper organization, \
flow, clarity, formatting)? Score 0.0 if primarily surface-level, 1.0 if deeply \
technical with minimal surface commentary. Reviews spending >40% on surface aspects \
should score below 0.4.

For each dimension, provide a brief justification (1-2 sentences), then a score.\
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_final_review(chat_history: list[dict]) -> str:
    """Extract the last assistant message content from chat history."""
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _read_paper_tex(task_path: str) -> str:
    """Read the paper's LaTeX source from the task directory."""
    tex_path = Path(task_path) / "latex" / "template.tex"
    if not tex_path.is_file():
        logger.warning(f"Paper tex not found at {tex_path}")
        return ""
    try:
        return tex_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Failed to read paper tex at {tex_path}: {e}")
        return ""


def count_external_citations(review_text: str) -> int:
    """Count distinct external paper references in the review text.

    Looks for:
    - arXiv IDs (e.g., 2301.12345, arXiv:2301.12345)
    - Author-year patterns (e.g., "Smith et al., 2023", "Smith & Jones (2022)")
    - DOIs
    """
    found = set()

    # arXiv IDs: 4-digit year prefix + 5-digit paper number (with optional version)
    for m in re.finditer(r'(?:arXiv:?\s*)?(\d{4}\.\d{4,5})(?:v\d+)?', review_text):
        found.add(f"arxiv:{m.group(1)}")

    # Author et al., YEAR or Author & Author (YEAR) or (Author et al., YEAR)
    for m in re.finditer(
        r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?)'
        r'[\s,]*[(\s]*'
        r'((?:19|20)\d{2})'
        r'[)\s]*',
        review_text,
    ):
        author_key = m.group(1).strip().lower()
        year = m.group(2)
        found.add(f"{author_key}_{year}")

    # DOIs
    for m in re.finditer(r'10\.\d{4,}/[^\s,)]+', review_text):
        found.add(f"doi:{m.group(0)}")

    return len(found)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# LLMJudgeReward callable class
# ---------------------------------------------------------------------------

class LLMJudgeReward:
    """Async-callable reward function that uses Gemini as a judge.

    Usage:
        reward_fn = LLMJudgeReward()
        reward = await reward_fn(chat_history, task_path)
    """

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        thinking_level: str = "low",
    ):
        if genai is None:
            raise ImportError(
                "google-genai is required for LLM judge rewards. "
                "Install with: pip install google-genai"
            )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is required for LLM judge rewards."
            )

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._thinking_level = thinking_level
        self._semaphore = asyncio.Semaphore(8)  # cap concurrent Gemini calls

        logger.info(
            f"LLMJudgeReward initialized: model={model}, thinking_level={thinking_level}"
        )

    async def __call__(self, chat_history: list[dict], task_path: str) -> float:
        """Compute reward by calling Gemini judge + local citation count.

        Returns float in [0.0, 1.0]. Falls back to format_reward on errors.
        """
        review_text = _extract_final_review(chat_history)
        if not review_text:
            logger.warning("LLM Judge: empty review, returning 0.0")
            return 0.0

        # Local rubric: citation count
        n_citations = count_external_citations(review_text)
        citation_score = _clamp(n_citations / 5.0)

        # Read paper tex for the LLM judge
        paper_tex = _read_paper_tex(task_path)
        if not paper_tex:
            logger.warning(
                "LLM Judge: paper tex unavailable, falling back to format reward"
            )
            return self._fallback_reward(chat_history, task_path, citation_score)

        # Build prompt
        prompt_text = JUDGE_PROMPT_TEMPLATE.format(
            paper_tex=paper_tex,
            review_text=review_text,
        )

        # Call Gemini with retry
        judge_output = await self._call_gemini(prompt_text)

        if judge_output is None:
            logger.warning("LLM Judge: Gemini call failed, falling back to format reward")
            return self._fallback_reward(chat_history, task_path, citation_score)

        # Compute weighted reward from LLM rubrics
        llm_reward = 0.0
        for rubric_name, weight in LLM_RUBRIC_WEIGHTS.items():
            rubric: RubricScore = getattr(judge_output, rubric_name)
            score = _clamp(rubric.score)
            llm_reward += weight * score

        final_reward = _clamp(llm_reward + CITATION_COUNT_WEIGHT * citation_score)

        # Log per-rubric breakdown
        logger.info(
            f"LLM Judge scores: "
            f"paper_spec={judge_output.paper_specificity.score:.2f} "
            f"intellectual={judge_output.intellectual_contribution.score:.2f} "
            f"evidence_gap={judge_output.evidence_claim_gap.score:.2f} "
            f"falsifiability={judge_output.falsifiability.score:.2f} "
            f"lit_pos={judge_output.literature_positioning.score:.2f} "
            f"tech_surface={judge_output.technical_vs_surface.score:.2f} "
            f"citations={n_citations}(score={citation_score:.2f}) "
            f"total={final_reward:.3f}"
        )

        return final_reward

    async def _call_gemini(self, prompt_text: str) -> LLMJudgeOutput | None:
        """Call Gemini with structured output, retry once on failure."""
        for attempt in range(2):
            try:
                async with self._semaphore:
                    response = await self._client.aio.models.generate_content(
                        model=self._model,
                        contents=prompt_text,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_json_schema=LLMJudgeOutput.model_json_schema(),
                            thinking_config=types.ThinkingConfig(
                                thinking_level=self._thinking_level,
                            ),
                            temperature=1.0,
                        ),
                    )
                return LLMJudgeOutput.model_validate_json(response.text)
            except Exception as e:
                logger.warning(
                    f"LLM Judge Gemini call failed (attempt {attempt + 1}/2): {e}"
                )
                if attempt == 0:
                    await asyncio.sleep(2)  # brief backoff before retry
        return None

    @staticmethod
    def _fallback_reward(
        chat_history: list[dict],
        task_path: str,
        citation_score: float,
    ) -> float:
        """Fall back to format_reward + citation score when Gemini is unavailable."""
        from ..paper_reviewer_generator import compute_format_reward

        format_r = compute_format_reward(chat_history, task_path)
        # Scale format_reward to occupy the LLM rubric weight portion
        llm_weight_total = sum(LLM_RUBRIC_WEIGHTS.values())
        return _clamp(
            llm_weight_total * format_r + CITATION_COUNT_WEIGHT * citation_score
        )
