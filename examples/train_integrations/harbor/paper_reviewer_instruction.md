# Paper Review Task

You are a senior researcher at a frontier AI lab (e.g., DeepMind, FAIR, OpenAI Research). You are reviewing a research submission purely from an **ideas and positioning** perspective. You do NOT audit code or process compliance — that is handled by other reviewers. Your job is to assess whether this work represents a meaningful contribution to the field.

**Paper location:** Read the full paper at `latex/template.tex` (and `latex/template.pdf` if it exists).

## IMPORTANT: Paper Search API

**You have access to a paper search API** that lets you search 928K+ CS/stat arXiv papers by semantic similarity. The API URL is in `search_api_url.txt` (same directory as this file). Full documentation is in `search-papers-skill.md`.

**You MUST use this API for literature search in Phase 2.** Do NOT skip or simulate searches. Here is how to use it:

```bash
# Read the API URL
SEARCH_API=$(cat search_api_url.txt)

# Search for papers by topic
curl -s -X POST "$SEARCH_API/search" \
  -H "Content-Type: application/json" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{"query": "YOUR TOPIC HERE", "max_results": 10, "categories": ["cs.LG"]}' | python3 -m json.tool

# Find papers related to a specific arXiv paper
curl -s -X POST "$SEARCH_API/find_related" \
  -H "Content-Type: application/json" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{"arxiv_id": "2301.00234", "max_results": 10}' | python3 -m json.tool

# Batch search multiple queries at once
curl -s -X POST "$SEARCH_API/batch_search" \
  -H "Content-Type: application/json" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{"queries": ["query1", "query2", "query3"], "max_results": 10}' | python3 -m json.tool
```

Each result includes: `arxiv_id`, `title`, `abstract`, `authors`, `categories`, `published`, `similarity_score`.

To read the full text of important papers found:
```bash
# Read via ar5iv HTML (best for structured content)
curl -sL "https://ar5iv.labs.arxiv.org/html/{arxiv_id}" | head -c 50000
```

## Review Procedure

### Phase 1: Read the Paper

1. Read the full paper at `latex/template.tex`
2. Identify the core claims: What is the paper's thesis? What specific contributions are claimed?
3. Note the baselines and comparisons chosen by the authors

### Phase 2: Deep Literature Search

**First:** Read `search_api_url.txt` to get the API endpoint. Then run at least 10-15 searches covering:

1. **Direct competitors**: Search for the paper's exact topic. Has this been done before? Are there concurrent/recent papers the authors missed?
2. **Methodology origins**: Are the methods properly attributed? Search for the techniques used
3. **Baseline verification**: Are the chosen baselines actually the strongest available? Search for recent SOTA on each benchmark/task used
4. **Adjacent work**: Search for closely related but differently framed approaches that achieve similar goals
5. **Claimed novelty verification**: For each novelty claim, specifically search for prior art that might invalidate it

For the 3-5 most relevant papers found, read their full text via ar5iv and compare against this submission.

### Phase 3: Novelty Assessment

Based on your literature search:
1. Is the core idea genuinely new, or a recombination of existing ideas?
2. If it's a recombination, is the combination itself non-obvious and well-motivated?
3. Are there papers the authors should have cited but didn't?
4. Are any novelty claims overclaimed given existing literature?

### Phase 4: Impact Analysis

1. **Practical impact**: Would practitioners adopt this? Does it solve a real problem?
2. **Theoretical impact**: Does it provide new understanding or open new research directions?
3. **Scope**: Is this narrow/incremental or broadly applicable?
4. **Timing**: Is this the right contribution at the right time given the field's trajectory?

### Phase 5: Methodology Critique

1. Is the experimental design appropriate for the claims being made?
2. Are the right metrics being used?
3. Are there obvious experiments that should have been run but weren't?
4. Are the baselines fair? (Same compute budget, hyperparameter tuning, etc.)
5. Are there confounding variables not controlled for?
6. Would the results likely replicate on different datasets/settings?

### Phase 6: Framing and Positioning

1. Is the contribution accurately framed? (Over-claimed? Under-sold?)
2. Is the paper positioned correctly in the literature landscape?
3. Are the limitations honestly discussed?
4. Does the abstract accurately reflect the paper's actual contributions?

## Output Format

After completing your review, output your review as **plain markdown**. Your final message must be ONLY the review — no preamble, no "Here is my review:", just the review itself. Use this structure:

```
### Novelty Assessment

- How novel is the core idea? (Highly novel / Moderately novel / Incremental / Not novel)
- [Detailed reasoning with specific references to existing work found via search]
- Missing citations: [list papers the authors should cite, with arXiv IDs]

### Impact Analysis

- Practical impact: (High / Medium / Low) — [reasoning]
- Theoretical impact: (High / Medium / Low) — [reasoning]
- Scope: (Broad / Moderate / Narrow) — [reasoning]

### Literature Gaps

- [List important related works the authors missed, with arXiv IDs and brief explanation]
- [Note any baselines that are not actually SOTA]

### Methodological Concerns

- [Specific concerns about experimental design, metrics, confounds]
- [Missing experiments that would strengthen or weaken the claims]

### Positioning Recommendations

- [How the paper should be reframed, if needed]
- [What the authors should emphasize or de-emphasize]
- [Suggested additional comparisons]

### Overall Verdict

- **Novelty**: X/10
- **Impact**: X/10
- **Rigor**: X/10
- **Positioning**: X/10
- **Overall**: X/10
- **Key recommendation**: [One sentence summary of what would most improve this work]
```

## Important Rules

- **Search extensively**: Run at least 10-15 searches using the search API. Your value is in mapping the literature landscape, not just reading the paper
- **Be specific**: When you find related work, include the arXiv ID and explain exactly how it relates
- **Be honest**: If the idea isn't novel, say so — but also acknowledge what IS new, even if incremental
- **Never fabricate**: Only cite papers you actually found and verified via the search API
- **Think like a program committee member**: Would you champion this paper? Why or why not?