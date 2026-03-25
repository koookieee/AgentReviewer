---
name: search-papers
description: Search for academic papers using arxiv-search-kit, OpenReview, and CrossRef. Use when you need to find related work, check novelty, gather citations, download/read full papers, or get BibTeX.
argument-hint: "[query or topic]"
---

# Academic Paper Search

Search query: $ARGUMENTS

Three tools, each with a clear role:

| Tool | Role | Auth |
|------|------|------|
| **arxiv-search-kit** | Fast semantic search, find related papers, batch search | None (local index, auto-downloads from HF) |
| **OpenReview** | Peer reviews, scores, BibTeX for ML venues | None needed for public data |
| **CrossRef** | BibTeX for any paper by DOI | None (use `mailto` for polite pool) |

**Setup** — install once:
```bash
pip install arxiv-search-kit[gpu]  # use [gpu] if CUDA available, else just: pip install arxiv-search-kit
```

---

## Decision Guide: Which Tool When?

| I want to... | Use |
|--------------|-----|
| Find papers on a topic | `client.search("query", categories=["cs.CV", "cs.LG"])` |
| Find papers related to a specific paper | `client.find_related("2301.00234", max_results=10)` |
| Search with context of current paper | `client.search("query", context_paper={"arxiv_id": "...", "title": "...", "abstract": "..."})` |
| Batch search multiple queries | `client.batch_search(["query1", "query2", ...])` |
| Look up a specific paper by ID | `client.get_paper("2301.00234")` |
| **Read full text of a paper** | **See "Reading Full Papers" below** |
| Read peer reviews + scores | OpenReview (`/notes?forum=<id>`) |
| Browse a venue's papers | OpenReview (`/notes?content.venueid=...`) |
| Get BibTeX | CrossRef via `dx.doi.org` (most reliable) |

---

## Searching with arxiv-search-kit

The search kit runs locally with a pre-built SPECTER2 index of 928K CS/stat papers. No API keys, no rate limits, 40ms per query on GPU.

```python
from arxiv_search_kit import ArxivClient

client = ArxivClient()  # auto-downloads index from HuggingFace on first use

# Basic search — returns papers ranked by semantic similarity
result = client.search("vision transformer object detection", max_results=10, categories=["cs.CV"])
for paper in result.papers:
    print(f"[{paper.arxiv_id}] {paper.title} ({paper.published.year})")
    print(f"  Score: {paper.similarity_score:.4f}")
    print(f"  Abstract: {paper.abstract[:200]}...")
```

### Context-aware search (recommended when reviewing a paper)

Pass the paper you're reviewing as context. This re-ranks results by similarity to your paper's domain, not just keyword match:

```python
result = client.search(
    "attention pruning efficiency",
    max_results=10,
    categories=["cs.LG", "cs.CV"],
    context_paper={
        "arxiv_id": "2301.00234",
        "title": "Efficient Vision Transformers via Token Pruning",
        "abstract": "We propose a method to prune redundant tokens..."
    }
)
```

### Find related papers by arXiv ID

```python
result = client.find_related("1706.03762", max_results=10)  # Attention Is All You Need
for paper in result.papers:
    print(f"[{paper.arxiv_id}] {paper.title} (score={paper.similarity_score:.4f})")
```

### Batch search (multiple queries at once)

```python
result = client.batch_search(
    ["vision transformer", "object detection real-time", "knowledge distillation"],
    max_results=10,  # per query
    categories=["cs.CV", "cs.LG"]
)
# Returns all unique papers across queries, deduplicated
```

### Get a specific paper's metadata

```python
paper = client.get_paper("2301.00234")
print(paper.title, paper.authors, paper.abstract, paper.categories)
```

### Using the arXiv ID to download papers

Every paper returned includes `paper.arxiv_id`. Use this to download:

```bash
# Download PDF
curl -s -L -o literature/2301.00234.pdf "https://arxiv.org/pdf/2301.00234"

# Read via ar5iv HTML (better for tables/math)
WebFetch(url="https://ar5iv.labs.arxiv.org/html/2301.00234", prompt="Extract methodology and results")
```

---

## Reading Full Papers

**You must read the full text of the 3-5 most important papers.** Abstracts alone are not sufficient for understanding methodology, experimental details, or making accurate comparisons.

**Always download papers to `literature/`** and record notes in `literature/README.md`:

```bash
mkdir -p literature
```

Use this priority chain:

### Priority 1: ar5iv HTML (best quality, most reliable)

All papers from arxiv-search-kit have arXiv IDs. Use the `paper.arxiv_id` field:

```
WebFetch(url="https://ar5iv.labs.arxiv.org/html/{arxiv_id}", prompt="Extract the full methodology, experimental setup, and key results")
```

ar5iv converts arXiv LaTeX sources to HTML. The content is structured, tables are preserved, and `WebFetch` can summarize exactly what you need in a single call.

**Also download the PDF** to `literature/` for figures and as a local backup:
```bash
curl -s -L -o literature/{arxiv_id}.pdf "https://arxiv.org/pdf/{arxiv_id}"
```

You can read the PDF with the `Read` tool (`pages` parameter, max 20 pages per call) if you need to inspect figures or verify details. **Rate limit:** Wait 3 seconds between consecutive arXiv downloads.

**Fallback — if ar5iv is unavailable**, read the downloaded PDF directly:
```
Read(file_path="literature/{arxiv_id}.pdf", pages="1-15")
```

### Priority 2: OpenReview (for non-arXiv venue papers)

If a paper is only on OpenReview (no arXiv ID), fetch it from there:
```bash
curl -s "https://api2.openreview.net/notes/search?term=PAPER+TITLE&source=forum&limit=1"
# Then use the PDF link from the response
```

### Priority 3: WebFetch the publisher page

```
WebFetch(url="https://doi.org/10.1234/example", prompt="Extract the full methodology, experimental setup, and key results")
```

### After reading each paper

Update `literature/README.md` with the paper's filename, title, and key takeaways.

---

## Quick Start: Common Workflows

### Literature search (find related work)

```python
from arxiv_search_kit import ArxivClient
client = ArxivClient()

# 1. Semantic search for your topic
result = client.search("contrastive learning self-supervised", max_results=10, categories=["cs.CV", "cs.LG"])

# 2. Find papers related to a key paper you already know
related = client.find_related("2002.05709", max_results=10)  # SimCLR

# 3. Broader search across multiple queries
batch = client.batch_search([
    "contrastive learning augmentation",
    "self-supervised representation learning",
    "momentum contrast",
], max_results=10)
```

### Get BibTeX for a paper

```bash
# From CrossRef via DOI (works for ALL DOI types)
curl -s -L -H "Accept: application/x-bibtex" "https://dx.doi.org/$DOI"
```

### Deep dive on a paper's review on OpenReview

```bash
# Search OpenReview for the paper
curl -s "https://api2.openreview.net/notes/search?term=PAPER+TITLE&source=forum&limit=1"

# Get reviews (use forum ID from above)
curl -s "https://api2.openreview.net/notes?forum=$FORUM_ID&limit=50"
```

---

## Search Tips

- **Use keyword phrases, not sentences** — "vision transformer pruning" works better than "how to prune vision transformers"
- **Specify categories** to narrow results: `categories=["cs.CV"]` for vision, `["cs.CL"]` for NLP
- **Use context_paper** when reviewing — it dramatically improves relevance by re-ranking around your paper's domain
- **Start broad, then narrow**: "regularization" then "L2 regularization neural networks"
- **Multiple searches**: 3-5 queries with different angles for coverage
- **Read full text**: After identifying key papers, always download and read them

## Gotchas

- **CrossRef `/transform`** fails for arXiv DOIs — always use `dx.doi.org`
- **OpenReview V2** wraps values: `content.title.value`, not `content.title`
- **OpenReview venue IDs** have no trailing slash: `ICLR.cc/2024/Conference`
- **arXiv rate limit** — wait 3 seconds between consecutive downloads