---
name: search
description: Search academic papers via arxiv-search-kit API and the web via Tavily CLI. Find related work, verify novelty, check baselines, get citations, read full papers, and search the web.
argument-hint: "[query or topic]"
---

# MANDATORY/IMPORTANT Search Tools

You have two search tools available:

1. **Paper Search API** — search 928K+ CS/stat arXiv papers, find related work, get citations, download LaTeX source. Uses the API URL in `search_api_url.txt`.
2. **Tavily CLI (`tvly`)** — web search and URL content extraction. Pre-installed and authenticated.

---

## Tool 1: Paper Search API

### Setup

```bash
SEARCH_API=$(cat search_api_url.txt)
```

### Primary Method: Batch Search with Importance Ranking

**This is the recommended method for literature search.** Searches multiple angles simultaneously, deduplicates, and ranks by relevance + citations + venue prestige.

```bash
curl -s -X POST "$SEARCH_API/batch_search" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "core topic of the paper you are reviewing",
      "specific method or technique used in the paper",
      "the problem domain or application area"
    ],
    "max_results": 15,
    "sort_by": "importance",
    "context_title": "Title of the paper you are reviewing",
    "context_abstract": "FULL ABSTRACT OF THE PAPER"
  }'
```

**Why this is best:**
- Multiple queries cover different angles (direct competitors, methods, applications)
- `context_title` + `context_abstract` bias results toward the paper's domain
- Results are deduplicated across queries

Each result includes: `arxiv_id`, `title`, `abstract`, `authors`, `year`, `similarity_score`, `citation_count`, `venue`, `tldr`.

Optional parameters: `categories` (e.g. `["cs.LG", "cs.CL"]`), `year`, `date_from`, `date_to`, `conference` (e.g. `"NeurIPS"`), `min_citations`.

### POST /search — Single query

```bash
curl -s -X POST "$SEARCH_API/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "vision transformer pruning",
    "max_results": 10,
    "sort_by": "importance",
    "categories": ["cs.CV", "cs.LG"],
    "context_title": "Paper Title",
    "context_abstract": "Abstract..."
  }'
```

### POST /find_related — Papers similar to a known paper

No query needed — finds nearest neighbors by embedding similarity.

```bash
curl -s -X POST "$SEARCH_API/find_related" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "1706.03762", "max_results": 10}'
```

### POST /citations — Papers that cite a given paper

```bash
curl -s -X POST "$SEARCH_API/citations" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "1706.03762", "limit": 50}'
```

### POST /references — Papers referenced by a given paper

```bash
curl -s -X POST "$SEARCH_API/references" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "1706.03762", "limit": 50}'
```

### POST /enrich — Add citation/venue data to papers

```bash
curl -s -X POST "$SEARCH_API/enrich" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_ids": ["1706.03762", "2010.11929"]}'
```

Returns: `citation_count`, `influential_citation_count`, `venue`, `tldr`, `publication_types`.

### POST /get_paper — Look up a single paper by arXiv ID

```bash
curl -s -X POST "$SEARCH_API/get_paper" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "1706.03762"}'
```

### POST /download_source — Download and extract a paper's LaTeX source

Step 1: Download and get the file listing:

```bash
curl -s -X POST "$SEARCH_API/download_source" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "1706.03762"}'
```

Response:
```json
{
  "arxiv_id": "1706.03762",
  "files": [
    {"path": "main.tex", "size_bytes": 45000},
    {"path": "references.bib", "size_bytes": 12000},
    {"path": "figures/arch.png", "size_bytes": 85000}
  ],
  "total_files": 3
}
```

Step 2: Read a specific file:

```bash
curl -s -X POST "$SEARCH_API/read_file" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "1706.03762", "file_path": "main.tex"}'
```

Response:
```json
{
  "arxiv_id": "1706.03762",
  "file_path": "main.tex",
  "content": "\\documentclass{article}\n...",
  "truncated": false,
  "size_bytes": 45000
}
```

Use `"max_chars": 50000` to limit large files. Default is 100K chars.

### Sort Options

| `sort_by` | What it does |
|-----------|-------------|
| `"relevance"` | Pure semantic + BM25 hybrid score (default) |
| `"importance"` | Relevance + citations + venue prestige (**recommended**) |
| `"citations"` | Most cited first |
| `"date"` | Newest first |

---

## Tool 2: Tavily Web Search (`tvly`)

Tavily is pre-installed and authenticated. Use it for **web search** — finding latest information about a topic, blog posts, research papers, project pages, or any web content.

### Search the web

```bash
tvly search "your search query"
```

Options:
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--depth` | `basic`, `advanced` | `advanced` | Search depth. `advanced` returns more detailed results |
| `--max-results` | `0-20` | `5` | Number of results |
| `--time-range` | `day`, `week`, `month`, `year` | — | Filter by time window |
| `--start-date` | `YYYY-MM-DD` | — | Results after this date |
| `--end-date` | `YYYY-MM-DD` | — | Results before this date |
| `--include-raw-content` | `markdown`, `text` | — | Include full page content |
| `--json` | flag | `false` | Output raw JSON |

Examples:

```bash
# Search for recent work on a topic
tvly search "process reward models for LLM training 2025"

# Get full page content for deeper reading
tvly search "RLHF alignment techniques" --include-raw-content markdown --max-results 3

# Search within a time range
tvly search "vision language models" --start-date 2025-01-01 --max-results 10

# JSON output for parsing
tvly search "neural architecture search" --json | python3 -m json.tool
```

### Extract content from a URL

When you find a relevant URL (from search results, paper references, or project pages), extract its content:

```bash
tvly extract https://example.com/article
```

Options:
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--query` | string | — | Rerank extracted chunks by relevance to this query |
| `--chunks-per-source` | `1-5` | — | Number of content chunks per URL (requires `--query`) |
| `--extract-depth` | `basic`, `advanced` | `basic` | `advanced` handles JavaScript-rendered pages |
| `--format` | `markdown`, `text` | `markdown` | Output format |

Examples:

```bash
# Extract content from a paper's project page
tvly extract https://github.com/some-project/repo

# Extract with relevance filtering
tvly extract https://docs.python.org/3/tutorial/ --query "list comprehensions" --chunks-per-source 3

# Extract multiple URLs at once (up to 20)
tvly extract https://example.com/page1 https://example.com/page2
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
    "max_results": 20,
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
  -d '{"arxiv_id": "FOUND_PAPER_ID", "max_results": 10}'
```

### Step 4: Check citation context

For key competing papers, check who cites them to find the latest work:

```bash
curl -s -X POST "$SEARCH_API/citations" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "KEY_COMPETITOR_ID", "limit": 30}'
```

### Step 5: Read full text of important papers

For the 3-5 most relevant papers, download their LaTeX source:

```bash
# Get file listing
curl -s -X POST "$SEARCH_API/download_source" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "PAPER_ID"}'

# Read the main .tex file
curl -s -X POST "$SEARCH_API/read_file" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "PAPER_ID", "file_path": "main.tex"}'
```

### Step 6: Web search for non-arXiv context or finding latest SOTA research/projects

Use Tavily heavily for things the paper search API can't find: SOTA in the current field of the paper you are revewing, blog posts, project pages, benchmarks:

```bash
# JUST AN EXAMPLE
tvly search "PROJECT_NAME benchmark results 2026"

# Read a project page or blog post
tvly extract https://github.com/org/project --query "methodology and results"
```

---

## When to Use Which Tool

| I want to... | Use |
|---|---|
| Find academic papers on a topic | Paper Search API `/batch_search` |
| Find papers related to a specific paper | Paper Search API `/find_related` |
| Check who cited a paper | Paper Search API `/citations` |
| Read a paper's LaTeX source | Paper Search API `/download_source` + `/read_file` |
| Get citation counts and venue info | Paper Search API `/enrich` |
| Search the web to find the SOTA in the field | `tvly search` |
| Find latest benchmarks and results | `tvly search` |
| Find latest paper/work in the field | `tvly search` |
| Learn more about the domain/topic of the paper | `tvly search` |
| Read content from a URL | `tvly extract` |
| Find blog posts, docs, project pages | `tvly search` |
| Find non-arXiv papers (OpenReview, etc.) | `tvly search` then `tvly extract` on the URL |

## Paper Search Tips

- **Use keyword phrases, not sentences for paper search API** — "vision transformer pruning" not "how to prune vision transformers"
- **Always pass `context_title` + `context_abstract`** — dramatically improves relevance
- **Use `sort_by: "importance"`** — surfaces influential papers, not just keyword matches
- **Filter by category** when needed: `"categories": ["cs.CV"]`
- **Filter by year/date** for recent work: `"year": 2024` or `"date_from": "2024-01-01"`
- **Use `min_citations`** to filter noise: `"min_citations": 5`
- **Extensively use `tavily`** to find the latest information, like SOTA in the field, latest papers/research projects, benchmarks, top performance on benchmarks