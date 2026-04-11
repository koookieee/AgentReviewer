---
name: search
description: Search academic papers via arxiv-search-kit API. Find related work, verify novelty, check baselines, get citations, query papers.
argument-hint: "[query or topic]"
---

# Search Tools

You have search tools available:

1. **Paper Search API** : search 928K+ CS/stat arXiv papers, find related work, get citations, query papers with natural language. API URL is in `/app/search_api_url.txt`.

---

## Tool: Paper Search API

### Setup: Do this ONCE at the start

```bash
SEARCH_API=$(cat /app/search_api_url.txt)
```

### Primary Method: Batch Search with Importance Ranking

**Recommended for literature search.** Searches multiple angles simultaneously, deduplicates, and ranks by relevance + citations + venue prestige. Uses Gemini embeddings — only queries are needed, no title/abstract context required.

```bash
curl -s -X POST "$SEARCH_API/batch_search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "core topic of the paper you are reviewing",
      "specific method or technique used",
      "the problem domain or application area"
    ],
    "max_results": 6,
    "sort_by": "importance"
  }'
```

**Why this is best:**
- Multiple queries cover different angles (direct competitors, methods, applications)
- Results are deduplicated across queries

Each result includes: `arxiv_id`, `title`, `abstract`, `citation_count`

Optional parameters: `categories` (e.g. `["cs.LG", "cs.CL"]`), `year`, `date_from`, `date_to`, `conference` (e.g. `"NeurIPS"`), `min_citations`.

### POST /query_paper — Ask anything about a paper

Unified endpoint for summarization, questions, or any natural language query about a paper. Downloads LaTeX source, converts to Markdown, and answers via Gemini.

Single paper:

```bash
curl -s -X POST "$SEARCH_API/query_paper" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "1706.03762", "query": "summarize this paper"}'
```

Multiple papers at once (parallel):

```bash
curl -s -X POST "$SEARCH_API/query_paper" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_ids": ["1706.03762", "2010.11929"], "query": "what are the key contributions?"}'
```

Response:
```json
{
  "responses": {
    "1706.03762": "...",
    "2010.11929": "..."
  }
}
```

Example queries: `"summarize this paper"`, `"what datasets were used?"`, `"explain the loss function"`, `"what are the ablation results?"`, `"what is the scaling factor in attention and why?"`.

Optional: `max_concurrent` (default 5) to control parallelism.

### POST /find_related — Find related papers

```bash
curl -s -X POST "$SEARCH_API/find_related" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "PAPER_ID", "max_results": 6}'
```

---

## Search Workflow for Paper Reviewing

### Step 1: Read the paper and identify key aspects

After reading the paper, identify:
- The core topic / thesis
- The specific methods or techniques used
- The problem domain or application area
- The baselines the authors compare against
- Each specific novelty claim

### Step 2: Run batch searches covering all angles

```bash
curl -s -X POST "$SEARCH_API/batch_search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "exact topic of the paper",
      "the main method or technique",
      "the application domain",
      "the key baseline method",
      "the claimed novelty in different words"
    ],
    "max_results": 6,
    "sort_by": "importance"
  }'
```

### Step 3: Drill into the most relevant papers

For the top 3-5 results, use `/find_related` to explore their neighborhood.

### Step 4: Query important papers

For the most relevant papers, get summaries or ask specific questions:

```bash
curl -s -X POST "$SEARCH_API/query_paper" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_ids": ["PAPER_ID_1", "PAPER_ID_2", "PAPER_ID_3"], "query": "summarize this paper"}'
```

---

## When to Use Which Endpoint

| I want to... | Use |
|---|---|
| Find academic papers on a topic | `/batch_search` |
| Find papers related to a specific paper | `/find_related` |
| Get a summary or ask anything about a paper | `/query_paper` |

## Paper Search Tips

- **Use keyword phrases, not sentences** — "vision transformer pruning" not "how to prune vision transformers"
- **Use `sort_by: "importance"`** — surfaces influential papers, not just keyword matches
- **Filter by category** when needed: `"categories": ["cs.CV"]`
- **Filter by year/date** for recent work: `"year": 2024` or `"date_from": "2024-01-01"`
- **Use `min_citations`** to filter noise: `"min_citations": 5`