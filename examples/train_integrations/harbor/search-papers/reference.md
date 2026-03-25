# API Reference — Academic Paper Search

## 1. arxiv-search-kit (Python)

Local semantic search over 928K CS/stat arXiv papers. No API keys, no rate limits.

```bash
pip install arxiv-search-kit[gpu]  # or without [gpu] for CPU-only
```

### 1a. Basic Search

```python
from arxiv_search_kit import ArxivClient

client = ArxivClient()  # auto-downloads index on first use (~4GB)

result = client.search(
    "vision transformer object detection",
    max_results=10,
    categories=["cs.CV", "cs.LG"],  # optional category filter
)
for paper in result.papers:
    print(f"[{paper.arxiv_id}] {paper.title} ({paper.published.year}) score={paper.similarity_score:.4f}")
```

**Parameters:**
- `query` — search terms
- `max_results` — number of results (default 20)
- `categories` — filter by arXiv categories (e.g. `["cs.CV", "cs.CL"]`)
- `date_from` / `date_to` — filter by date (ISO format strings)
- `rerank` — enable graph-based re-ranking (default True)
- `context_paper` — dict with `arxiv_id`, `title`, `abstract` of paper being reviewed

### 1b. Context-Aware Search

```python
result = client.search(
    "attention pruning efficiency",
    max_results=10,
    context_paper={
        "arxiv_id": "2301.00234",
        "title": "Efficient Vision Transformers via Token Pruning",
        "abstract": "We propose a method to prune redundant tokens..."
    }
)
```

### 1c. Find Related Papers

```python
result = client.find_related("1706.03762", max_results=10)
```

### 1d. Batch Search

```python
result = client.batch_search(
    ["vision transformer", "object detection", "knowledge distillation"],
    max_results=10,  # per query
    categories=["cs.CV"]
)
# Returns all unique papers across queries, deduplicated
```

### 1e. Get Paper by ID

```python
paper = client.get_paper("1706.03762")
print(paper.title, paper.authors, paper.abstract, paper.categories)
```

### Paper Object Fields

| Field | Type | Description |
|-------|------|-------------|
| `arxiv_id` | str | ArXiv paper ID (e.g. `"1706.03762"`) |
| `title` | str | Paper title |
| `abstract` | str | Full abstract |
| `authors` | list[str] | Author names |
| `categories` | list[str] | ArXiv categories |
| `primary_category` | str | Primary category (e.g. `"cs.CV"`) |
| `published` | datetime | Publication date |
| `updated` | datetime | Last updated date |
| `similarity_score` | float | Relevance score (0-1, higher = more relevant) |
| `citation_count` | int | Citation count (after enrichment) |

---

## 2. arXiv Direct Download

No API key needed. Use `paper.arxiv_id` from search results.

### 2a. Download PDF

```bash
mkdir -p literature
curl -s -L -o literature/1706.03762.pdf "https://arxiv.org/pdf/1706.03762"
```

Then read with Claude's `Read` tool:
```
Read(file_path="literature/1706.03762.pdf", pages="1-15")
```

### 2b. Read HTML Version

ar5iv (broadest coverage):
```
WebFetch(url="https://ar5iv.labs.arxiv.org/html/1706.03762", prompt="Extract the methodology and results")
```

arXiv native HTML (recent papers only):
```
WebFetch(url="https://arxiv.org/html/1706.03762v7", prompt="Extract the methodology and results")
```

### arXiv Rate Limits

- Wait at least 3 seconds between consecutive downloads
- Bulk automated downloads are prohibited — only download papers you need to read

---

## 3. OpenReview

Base URL: `https://api2.openreview.net`

No auth needed for public data. Covers ICLR (2013+), NeurIPS (2019+), ICML (2023+), TMLR, workshops.

### 3a. Keyword Search

```bash
curl -s "https://api2.openreview.net/notes/search?term=in-context+learning&source=forum&limit=10"
```

**Parameters:**
- `term` — search keywords
- `source` — `forum` (papers), `reply` (reviews/comments), `all`
- `limit` — max results (up to 1000)

### 3b. Venue-Specific Query

```bash
curl -s "https://api2.openreview.net/notes?content.venueid=ICLR.cc/2024/Conference&limit=50&select=id,forum,content.title,content.authors,content.abstract,content.venue,content.keywords"
```

Venue IDs: `ICLR.cc/2024/Conference`, `NeurIPS.cc/2024/Conference`, `ICML.cc/2024/Conference` (no trailing slash).

### 3c. Get Reviews

```bash
curl -s "https://api2.openreview.net/notes?forum=$FORUM_ID&limit=50" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for n in d.get('notes', []):
    invs = n.get('invitations', [])
    c = n.get('content', {})
    if any('Official_Review' in i for i in invs):
        rating = c.get('rating', {}).get('value', '?') if isinstance(c.get('rating'), dict) else c.get('rating', '?')
        confidence = c.get('confidence', {}).get('value', '?') if isinstance(c.get('confidence'), dict) else c.get('confidence', '?')
        print(f'Review: rating={rating} confidence={confidence}')
    elif any('Decision' in i for i in invs):
        decision = c.get('decision', {}).get('value', '?') if isinstance(c.get('decision'), dict) else c.get('decision', '?')
        print(f'Decision: {decision}')
"
```

### 3d. Download PDF

```bash
curl -s "https://api2.openreview.net/pdf?id=<NOTE_ID>" -o paper.pdf
```

### 3e. Extract Content Fields

V2 wraps values: `content.title.value`, not `content.title`.

```bash
curl -s "https://api2.openreview.net/notes?content.venueid=ICLR.cc/2024/Conference&limit=1" | python3 -c "
import json, sys
d = json.load(sys.stdin)
n = d['notes'][0]; c = n['content']
print(f'Title: {c[\"title\"][\"value\"]}')
print(f'Authors: {c[\"authors\"][\"value\"]}')
print(f'Venue: {c[\"venue\"][\"value\"]}')
print(f'URL: https://openreview.net/forum?id={n[\"forum\"]}')
"
```

### OpenReview Rate Limits

- No published limits. 5 concurrent requests OK.
- Max 1000 items/request. Use offset for pagination.

---

## 4. CrossRef

### 4a. BibTeX via dx.doi.org

Works for ALL DOI types (arXiv, ACL, ACM, Springer, etc.).

```bash
curl -s -L -H "Accept: application/x-bibtex" "https://dx.doi.org/10.48550/arXiv.1810.03292"
curl -s -L -H "Accept: application/x-bibtex" "https://dx.doi.org/10.18653/v1/N19-1423"
```

**Important:** Use `dx.doi.org`, NOT `api.crossref.org/works/{doi}/transform`.

### 4b. Keyword Search (find DOI by title)

```bash
curl -s "https://api.crossref.org/works?query=attention+is+all+you+need&rows=3&mailto=test@example.com&select=DOI,title,author,is-referenced-by-count"
```

### CrossRef Rate Limits

- `mailto=email` enables polite pool (~50 req/s). 5 concurrent OK.
