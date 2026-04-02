# LLM-as-Judge: Review Quality Evaluation

You are an experienced area chair at a top-tier ML conference. You are evaluating the quality of a peer review written by a reviewer.

You will be given:
1. The **title and abstract** of the paper under review
2. **1-2 human reviews** of the same paper (these are high-quality reference reviews from real reviewers)
3. The **model's review** (the review you are evaluating)

Your job is to grade the model's review on 5 criteria. For each criterion, read the definition carefully, then output your score.

---

## Paper Information

**Title:** {title}

**Abstract:**
{abstract}

---

## Human Reviews (Reference)

{human_reviews}

---

## Model's Review (To Evaluate)

{model_review}

---

## Evaluation Criteria

### Criterion 1: Comprehension

Does the reviewer actually understand what this paper is about?

Read the title, abstract, and the model's review. Ask yourself: does this review correctly identify the paper's core contribution, the method proposed, and the claims being made? Or does the review mischaracterize the paper — confuse the method with something else, critique something the paper never claimed, or describe a different paper entirely?

This is not about whether the reviewer agrees with the paper. It is about whether they understood it.

**Score 1** if the review demonstrates that the reviewer understood the paper's core idea, method, and claims.
**Score 0** if the review mischaracterizes the contribution, confuses the method, critiques claims the paper didn't make, or shows no evidence of understanding beyond the title.

### Criterion 2: Substance and Specificity

Does the review engage with technical content in a way that is specific, grounded, and non-trivial?

A review can fail this criterion in three ways:
- It talks primarily about surface features: organization, writing quality, figure clarity, paper structure, motivation phrasing. These observations are not worthless, but a review dominated by them is not substantive.
- It makes technical-sounding points that are vague or generic: "the baselines are weak", "needs more experiments", "the method is incremental". These give the author nothing to work with.
- It raises technical points that are trivially obvious — questions anyone could ask from reading just the abstract, without engaging with the actual methods or results. For example, "Why didn't you use a different loss function?" with no context for why that would matter for this specific method.

A good review passes all three bars: it is technical (about methods, results, claims, evidence), it is specific (the author knows exactly what is being referred to and could act on it), and it is non-trivial (the points reflect genuine engagement with the paper's actual content).

**Score 1** if the review raises technical points that are specific AND non-trivial — reflecting real engagement with the paper's methods, results, or experimental design.
**Score 0** if the review is primarily about presentation/structure, OR the technical points are vague/generic, OR the technical points are trivially obvious.

### Criterion 3: Insight

Does the review contain at least one observation that goes beyond what is obvious from the abstract?

This is the hardest criterion to pass. Ask yourself: is there anything in this review — a concern, a connection, a question, a limitation, a suggested experiment — that you could NOT have written from reading only the title and abstract? Something that requires having actually thought about the paper's details, its methodology, its results, or its relationship to existing work?

A review that simply paraphrases the abstract's claims and adds generic praise or criticism fails this criterion. A review that identifies a specific methodological concern, notices a gap between claims and evidence, connects the work to a non-obvious related area, or raises a question that only makes sense if you've read the actual paper — that passes.

**Score 1** if the review contains at least one non-obvious observation that could only come from engaging with the paper's actual content beyond the abstract.
**Score 0** if everything in the review could have been written by someone who only read the title and abstract.

### Criterion 4: Issue Overlap

Did the model's review catch the same important issues that the human reviewers caught?

Compare the model's review against the human review(s) section by section:
- **Strengths**: Did the model identify similar strengths as the humans?
- **Weaknesses**: Did the model catch similar weaknesses or concerns?
- **Questions**: Did the model raise similar questions or flag similar gaps?

Focus on substantive issues, not wording. Two reviewers can describe the same concern in completely different words — what matters is whether they noticed the same underlying issue.

**Scoring (continuous, 0.0 to 1.0):**

If 1 human review is provided:
- Identify the human reviewer's major points across strengths, weaknesses, and questions.
- Score based on what fraction of these major points the model's review also covers (in substance, not wording).
- Full coverage of major points → 1.0. No overlap at all → 0.0.

If 2 human reviews are provided:
- Identify the major points from each human reviewer.
- Points that BOTH human reviewers raise are convergent issues — these are almost certainly real and important. Weight these higher.
- Score based on coverage: covering convergent issues from both reviewers scores higher than covering issues raised by only one.
- Full coverage of both reviewers' major points → 1.0. Only covers one reviewer's points partially → 0.3-0.5. No overlap → 0.0.

### Criterion 5: Calibration

Is the model's overall judgment of the paper in the right ballpark compared to human reviewers?

Compare the model's scores and decision against the human reviewers'. The most important dimensions to compare are:

- **Overall score** and **Contribution score** — these reflect the reviewer's judgment about the paper's idea and its value to the field. Getting these right is the hardest and most important part of reviewing.
- **Soundness score** — reflects technical judgment about whether the paper's claims are supported.
- **Decision** (Accept/Reject) — the bottom line.

Do NOT weight Presentation score heavily for calibration — it measures surface quality, not the paper's scientific merit. Do NOT use Confidence score — it is meta-information about the reviewer, not about the paper.

**Scoring:**
- **1.0**: The model's Overall and Contribution scores are within reasonable range of the human average (±2 on their respective scales), AND the model's Accept/Reject decision matches the human consensus.
- **0.5**: The model's scores trend in the right direction but are off by more than 2 points, OR the decision disagrees with humans but the scores are close.
- **0.0**: The model's assessment is dramatically miscalibrated — for example, humans say strong accept with high scores and the model says reject with low scores, or vice versa.

---

## Output Format

Return your evaluation as JSON with this exact structure:

```json
{
  "comprehension": {
    "justification": "...",
    "score": 0 or 1
  },
  "substance_and_specificity": {
    "justification": "...",
    "score": 0 or 1
  },
  "insight": {
    "justification": "...",
    "score": 0 or 1
  },
  "issue_overlap": {
    "justification": "...",
    "score": 0.0 to 1.0
  },
  "calibration": {
    "justification": "...",
    "score": 0.0 or 0.5 or 1.0
  }
}
```

For each criterion, write the justification FIRST (1-2 sentences explaining your reasoning), then the score. Be strict — a mediocre review should not pass the standalone criteria just because it avoids being terrible.