# arxiv-search-kit

Offline ArXiv paper search over 928K CS papers. Two embedding backends, LanceDB vector index + BM25 hybrid retrieval.

**SPECTER2**: 40ms per search on GPU. No API keys required. No rate limits.

**Gemini-2**: Higher quality semantic search via `gemini-embedding-2-preview` (3072-dim). Requires a Gemini API key.

## Install

### pip

```bash
pip install arxiv-search-kit[cpu]
```

### uv

```bash
uv pip install arxiv-search-kit[cpu]
# or in a project
uv add "arxiv-search-kit[cpu]"
```

### GPU (CUDA)

PyTorch with CUDA must be installed separately.

```bash
# install CUDA torch first (pick your CUDA version: https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# then install the kit
pip install arxiv-search-kit[gpu]
```

With uv:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install arxiv-search-kit[gpu]
```

> The pre-built index (~4GB) auto-downloads from HuggingFace on first use.

## Quick Start

```python
from arxiv_search_kit import ArxivClient

# SPECTER2 (default) — local, no API key needed
client = ArxivClient()

# Gemini-2 — higher quality, requires Gemini API key
client = ArxivClient(embedding="gemini", gemini_api_key="AIza...")
# or set GEMINI_API_KEY env var and omit gemini_api_key

results = client.search("attention mechanism transformers")
for paper in results:
    print(paper.title, paper.arxiv_id)
```

## Default vs Extra Fields

By default, search results include only the essential fields:
**arxiv_id, title, abstract, citation_count** (citation_count populated only when using `sort_by="importance"` or `sort_by="citations"`).

Pass `details="extra"` to get all fields.

```python
# default — only core fields
results = client.search("transformers")
results.to_dicts()
# [{"arxiv_id": "...", "title": "...", "abstract": "...", "citation_count": None}, ...]

# extra — all fields (authors, categories, doi, venue, tldr, etc.)
results = client.search("transformers", details="extra")
results.to_dicts()
# [{"arxiv_id": "...", "title": "...", "authors": [...], "categories": [...], ...}, ...]
```

Works the same for `batch_search` and `find_related`:

```python
results = client.batch_search(["BERT", "GPT"], sort_by="importance", details="extra")
related  = client.find_related("1706.03762", details="extra")
```

## Search

### Keyword Search

```python
# basic search
results = client.search("vision transformers object detection", max_results=20)

# filter by category, year, or date range
results = client.search("graph neural networks", categories=["cs.LG", "cs.AI"], year=2024)
results = client.search("LLM safety", date_from="2024-01-01", date_to="2024-06-30")

# conference-aware search (maps conference name to ArXiv categories)
results = client.search("object detection", conference="CVPR", year=2024)
```

### Context-Biased Search

Bias results toward a specific paper's neighborhood — useful when searching for related work.

```python
# by ArXiv ID (uses stored embedding from the index)
results = client.search("self-supervised learning", context_paper_id="2010.11929")

# by title + abstract (embeds on the fly)
results = client.search(
    "sim-to-real transfer",
    context_title="My Paper Title",
    context_abstract="We propose a method for...",
)
```

### Batch Search

Run multiple queries covering different angles of a topic, merge and deduplicate results.

```python
results = client.batch_search(
    queries=[
        "reinforcement learning from human feedback",
        "process reward model RLHF",
        "on-policy distillation language model",
    ],
    max_results=15,
    context_title="My Paper Title",
    context_abstract="...",
)
```

### Sorting & Importance Ranking

```python
# sort by relevance (default) — pure semantic + BM25 hybrid score
results = client.search("diffusion models", sort_by="relevance")

# sort by citation count (calls Semantic Scholar API)
results = client.search("diffusion models", sort_by="citations")

# sort by date
results = client.search("diffusion models", sort_by="date")

# sort by importance — blends relevance with citation count,
# venue prestige, and influential citation ratio via S2 API.
# Surfaces the most relevant *important* papers.
results = client.search("diffusion models", sort_by="importance")

# filter by minimum citations
results = client.search("transformers", min_citations=50)
```

`sort_by="importance"` works with both `search()` and `batch_search()`:

```python
results = client.batch_search(
    queries=["query angle 1", "query angle 2", "query angle 3"],
    max_results=15,
    sort_by="importance",
    context_title="Your Paper Title",
    context_abstract="Your abstract...",
)
```

### Find Related Papers

Find papers similar to a given paper using its stored embedding. No keyword query needed.

```python
related = client.find_related("1706.03762", max_results=10)  # Attention Is All You Need
related = client.find_related("1706.03762", categories=["cs.CL"])  # filter by category
```

## Paper Object

Every search returns a `SearchResult` containing `Paper` objects:

```python
paper = results[0]

# Core fields (from ArXiv metadata)
paper.arxiv_id          # "2401.12345"
paper.title             # "Paper Title"
paper.abstract          # "We propose..."
paper.authors           # [Author(name="Alice", affiliation="MIT"), ...]
paper.categories        # ["cs.CV", "cs.LG"]
paper.primary_category  # "cs.CV"
paper.published         # datetime(2024, 1, 15)
paper.updated           # datetime(2024, 3, 1)
paper.doi               # "10.1234/..." or None
paper.journal_ref       # "NeurIPS 2024" or None
paper.comment           # "Accepted at..." or None

# Computed
paper.pdf_url           # "https://arxiv.org/pdf/2401.12345"
paper.abs_url           # "https://arxiv.org/abs/2401.12345"
paper.year              # 2024
paper.first_author      # "Alice"
paper.author_names      # ["Alice", "Bob"]

# Search score
paper.similarity_score  # 0.87 (set after search)

# Enrichment fields (populated after client.enrich() or sort_by="importance")
paper.citation_count           # 142
paper.influential_citation_count  # 23
paper.venue                    # "Neural Information Processing Systems"
paper.publication_types        # ["Conference"]
paper.references               # ["1706.03762", ...] (ArXiv IDs)
paper.tldr                     # "This paper proposes..."

# Serialization
paper.to_dict()         # dict with all fields
paper.to_bibtex()       # BibTeX string
paper.to_bibtex("acl")  # ACL-style BibTeX
```

`SearchResult` supports `len()`, iteration, and indexing:

```python
results = client.search("transformers")
len(results)         # 20
results[0]           # first Paper
results.query        # "transformers"
results.search_time_ms  # 42.5
```

## Semantic Scholar Enrichment

Enrich papers with citation data, venue info, and AI-generated summaries via the [Semantic Scholar API](https://api.semanticscholar.org/api-docs/graph).

```python
# enrich search results
results = client.search("attention mechanism")
client.enrich(results)

results[0].citation_count   # 95421
results[0].venue             # "Neural Information Processing Systems"
results[0].tldr              # "A new architecture based solely on attention..."

# enrich specific fields only
client.enrich(results, fields=["citationCount", "venue"])
```

### Citation Graph

```python
# papers that cite this paper
citations = client.get_citations("1706.03762", limit=100)
# [{"arxiv_id": "...", "title": "...", "year": 2023, "citation_count": 42}, ...]

# papers referenced by this paper
references = client.get_references("1706.03762", limit=100)
```

### Rate Limits

The S2 API works without a key (5,000 requests / 5 minutes shared pool). For heavier use, set `S2_API_KEY`:

```bash
export S2_API_KEY=your_key_here
```

## Download Papers

Download PDFs or LaTeX source archives directly from ArXiv.

```python
# single paper — by ID or Paper object
path = client.download_pdf("1706.03762", output_dir="./papers")
path = client.download_source("1706.03762", output_dir="./sources")

# from search results
results = client.search("vision transformers", max_results=5)
paths = client.download_papers(results.papers, output_dir="./papers", format="pdf")
paths = client.download_papers(results.papers, output_dir="./sources", format="source")
```

Downloads are streamed to disk (no full file in memory). Failed downloads are skipped with a warning.

## Paper Q&A

Query any paper with natural language — summarization, specific questions, anything. Downloads the LaTeX source from ArXiv, converts it to clean Markdown via pandoc (preserving equations and tables as raw LaTeX blocks), then passes your query + the paper content to Gemini.

```bash
pip install arxiv-search-kit[summarize]
# also requires pandoc: apt install pandoc
```

```python
# summarize
response = client.query_paper("1706.03762", "summarize this paper")

# ask a specific question
response = client.query_paper("1706.03762", "What is the scaling factor in the attention mechanism and why is it used?")

# any natural language — the LLM handles intent
response = client.query_paper("1706.03762", "give me a tldr")
response = client.query_paper("1706.03762", "what datasets were used and how did they split them?")
response = client.query_paper("1706.03762", "explain the loss function")

# from a Paper object
results = client.search("vision transformers", max_results=1)
response = client.query_paper(results[0], "what are the key contributions?")

# set env var to avoid passing api_key every time
# export GEMINI_API_KEY=your-gemini-key
response = client.query_paper("1706.03762", "summarize")
```

### Batch — parallel across multiple papers

Pass a list to query multiple papers in parallel. Returns a dict mapping ArXiv ID to response.

```python
results = client.search("vision transformers", max_results=5)
responses = client.query_paper(results.papers, "summarize this paper")
# {"2401.12345": "...", "2312.67890": "...", ...}

# control parallelism (default: 5 concurrent)
responses = client.query_paper(results.papers, "what datasets were used?", max_concurrent=3)
```

## Async Support

All main methods have async variants:

```python
results = await client.async_search("transformers", max_results=10)
results = await client.async_batch_search(queries=[...], sort_by="importance")
related = await client.async_find_related("1706.03762")
await client.async_enrich(results)
response = await client.async_query_paper("1706.03762", "summarize this paper")
response = await client.async_query_paper(results.papers, "what optimizer was used?")
```

## Venue Prestige Tiers

When using `sort_by="importance"`, papers are scored by a combination of citation count, influential citation ratio, and venue prestige. Venues are assigned tiers:

| Tier | Weight | Venues |
|------|--------|--------|
| 3 (top) | 1.0 | NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR, ICCV, ECCV, AAAI, KDD, JMLR, TPAMI, ... |
| 2 (strong) | 0.67 | WACV, COLING, ICRA, SIGGRAPH, AISTATS, COLT, Findings, ... |
| 1 (decent) | 0.33 | BMVC, ACCV, SemEval, CoNLL, ... |
| 0 (unknown) | 0.0 | ArXiv-only, unrecognized venues |

The importance score formula:

```
importance = 0.55 * log_citation_score + 0.30 * venue_score + 0.15 * influential_ratio
final_score = 0.6 * relevance + 0.4 * importance
```

## Coverage

928K papers across all major CS + stat.ML + eess categories:

- **cs.CV** (144K), **cs.LG** (129K), **cs.CL** (78K), **cs.AI** (36K), **cs.RO** (38K), **cs.CR** (32K), **stat.ML** (20K), and 40+ more subcategories.

Conference-to-category mappings: CVPR, NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, AAAI, IJCAI, CHI, KDD, SIGIR, RSS, ICRA, and [many more](arxiv_search_kit/categories.py).

## Embedding Backends

### SPECTER2 (default)
- Local transformer model ([allenai/specter2](https://huggingface.co/allenai/specter2)), 768-dim embeddings
- No API key, no rate limits, ~40ms on GPU
- Best with `context_title` + `context_abstract` for keyword queries
- Index: [anonymousatom/arxiv-search-index](https://huggingface.co/datasets/anonymousatom/arxiv-search-index) (~4GB)

### Gemini-2
- `gemini-embedding-2-preview` via Google Gemini API, 3072-dim embeddings
- Requires a Gemini API key (`gemini_api_key=` or `GEMINI_API_KEY` env var)
- Works with just a query string — no title or abstract needed
- Better out-of-the-box quality for keyword queries — asymmetric retrieval format handles free-text queries natively
- `batch_search` pre-embeds all queries in a single API call (not N separate calls)
- Index: [Vidushee/arxiv-gemini-index](https://huggingface.co/datasets/Vidushee/arxiv-gemini-index) (~10GB)

```python
# SPECTER2 — fast, local, no key needed
client = ArxivClient()
results = client.search("attention mechanism transformers")  # works fine

# SPECTER2 benefits from context when you have it
results = client.search(
    "attention mechanism transformers",
    context_title="My Paper Title",
    context_abstract="We propose...",
)

# Gemini-2 — higher quality, just a query is enough
client = ArxivClient(embedding="gemini", gemini_api_key="AIza...")
results = client.search("attention mechanism transformers")
results = client.batch_search(["BERT", "GPT", "RLHF"])  # single API call for all queries
```

## How It Works

1. **Index**: 928K papers embedded with SPECTER2 or Gemini-2, stored in [LanceDB](https://lancedb.github.io/lancedb/)
2. **Retrieval**: Hybrid search — dense (cosine) + sparse (BM25) fused via Reciprocal Rank Fusion
3. **Re-ranking**: Personalized PageRank on a k-NN similarity graph built from candidate embeddings
4. **Enrichment**: Optional citation/venue data from [Semantic Scholar API](https://api.semanticscholar.org/api-docs/graph)
5. **Importance**: Blends relevance with citation count, venue prestige, and influential citation ratio

Indexes auto-download from HuggingFace on first use.

## Building Your Own Index

Only needed if you want to customize the paper set or update to the latest papers.

```bash
pip install arxiv-search-kit[index]

# download metadata from ArXiv OAI-PMH (~2 hours, network-bound)
python -m arxiv_search_kit.scripts.build_index download --output metadata.jsonl

# build index (needs GPU, ~45 min)
python -m arxiv_search_kit.scripts.build_index build \
    --metadata-path metadata.jsonl \
    --output-dir ./my_index \
    --device cuda

# or do both in one step
python -m arxiv_search_kit.scripts.build_index all --output-dir ./my_index --device cuda
```

Then point the client to your custom index:

```python
client = ArxivClient(index_dir="./my_index", device="cuda")
```

## License

MIT
