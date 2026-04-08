---
name: search
description: Search academic papers via arxiv-search-kit API. Find related work, verify novelty, check baselines, get citations, get paper summaries.
argument-hint: "[query or topic]"
---

# MANDATORY/IMPORTANT Search Tools

You have a search tools available:

1. **Paper Search API** : search 928K+ CS/stat arXiv papers, find related work, get citations, get comprehensive paper summaries. API URL is in `/app/search_api_url.txt`.

---

## Tool: Paper Search API

### Setup: Do this ONCE at the start, then reuse $SEARCH_API and $PAPER_TITLE/$PAPER_ABSTRACT in every call. You should save the SEARCH_API, PAPER_TITLE AND PAPER_ABSTRACT at start and use them through variables, not typing again completely.

```bash
# Store the search_api, paper title and abstract you extracted so you can directly use later by using varaibles names and not typing again completely:

SEARCH_API=$(cat /app/search_api_url.txt)
PAPER_TITLE="<title of the paper under review>"
PAPER_ABSTRACT="<full abstract of the paper under review>"
```

### Primary Method: Batch Search with Importance Ranking

**This is the recommended method for literature search.** Searches multiple angles simultaneously, deduplicates, and ranks by relevance + citations + venue prestige.

> **IMPORTANT: ALWAYS include `context_title` and `context_abstract` which you have saved, in EVERY search call**, biases results toward the paper's domain and dramatically improves relevance. Never omit them.

```bash
curl -s -X POST "$SEARCH_API/batch_search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "core topic of the paper you are reviewing",
      "specific method or technique used in the paper",
      "the problem domain or application area"
    ],
    "max_results": 6, # Preferred 6 for not bloating the context
    "sort_by": "importance",
    "context_title": "'"$PAPER_TITLE"'",
    "context_abstract": "'"$PAPER_ABSTRACT"'"
  }'
```

**Why this is best:**
- Multiple queries cover different angles (direct competitors, methods, applications)
- `context_title` + `context_abstract` bias results toward the paper's domain, **always pass these**
- Results are deduplicated across queries

Each result includes: `arxiv_id`, `title`, `abstract`, `year`, `citation_count`

Optional parameters: `categories` (e.g. `["cs.LG", "cs.CL"]`), `year`, `date_from`, `date_to`, `conference` (e.g. `"NeurIPS"`), `min_citations`.


### POST /summarize_paper, Get a comprehensive summary of a paper

**This is the recommended way to read related papers.** Returns a detailed summary covering the paper's contributions, methods, results, and limitations, without downloading the full source.

Single paper:

```bash
curl -s -X POST "$SEARCH_API/summarize_paper" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "1706.03762"}'
```

Multiple papers at once (parallel):

```bash
curl -s -X POST "$SEARCH_API/summarize_paper" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_ids": ["1706.03762", "2010.11929"]}'
```

Response:
```json
{
  "summaries": {
    "1706.03762": "Comprehensive summary of Attention Is All You Need...",
    "2010.11929": "Comprehensive summary of ViT..."
  }
}
```

Optional parameters: `max_concurrent` (default 5) to control parallelism.

## POST /ask_paper, Ask any question about a paper and get an answer grounded in the paper's content.

Use when you need a specific answer from a paper — exact method details, ablation results, experimental setup — rather than a full summary.

```bash
curl -s -X POST "$SEARCH_API/ask_paper" \
  -H "Content-Type: application/json" \
  -d '{
    "arxiv_id": "1706.03762",
    "question": "What are the key contributions of this paper?"
  }'
```

Response:
```json
{
  "arxiv_id": "1706.03762",
  "question": "...",
  "answer": "..."
}
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

Run a batch search with 5-8 queries covering different angles:

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
    "max_results": 6, # Preferred 6 for not bloating the context
    "sort_by": "importance",
    "context_title": "...",
    "context_abstract": "..."
  }'
```

### Step 3: Drill into the most relevant papers

For the top 3-5 results, use `/find_related` to explore their neighborhood:

```bash
curl -s -X POST "$SEARCH_API/find_related" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "FOUND_PAPER_ID", "max_results": 6}'
```


### Step 4: Get summaries of important papers

For the 3-5 most relevant papers, get comprehensive summaries:

```bash
# Summarize multiple papers at once
curl -s -X POST "$SEARCH_API/summarize_paper" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_ids": ["PAPER_ID_1", "PAPER_ID_2", "PAPER_ID_3"]}'
```


---

## POST /ask_paper: Ask any question about a paper, grounded in its full content

Use when you need a specific answer from a paper — exact method details, ablation results, experimental setup — rather than a full summary.

```bash
curl -s -X POST "$SEARCH_API/ask_paper" \
  -H "Content-Type: application/json" \
  -d '{
    "arxiv_id": "1706.03762",
    "question": "What are the key contributions of this paper?"
  }'
```

Response:
```json
{
  "arxiv_id": "1706.03762",
  "question": "...",
  "answer": "..."
}
```

Ask really good questions.


---

## When to Use Which Tool

| I want to... | Use |
|---|---|
| Find academic papers on a topic | Paper Search API `/batch_search` |
| Find papers related to a specific paper | Paper Search API `/find_related` |
| Get a full overview of a paper | Paper Search API `/summarize_paper` |
| Get a specific answer from a paper | Paper Search API `/ask_paper` |

## Paper Search Tips

- **Use keyword phrases, not sentences for paper search API** — "vision transformer pruning" not "how to prune vision transformers"
- **Always pass `context_title` + `context_abstract`** — dramatically improves relevance
- **Use `sort_by: "importance"`** — surfaces influential papers, not just keyword matches
- **Filter by category** when needed: `"categories": ["cs.CV"]`
- **Filter by year/date** for recent work: `"year": 2024` or `"date_from": "2024-01-01"`
- **Use `min_citations`** to filter noise: `"min_citations": 5`