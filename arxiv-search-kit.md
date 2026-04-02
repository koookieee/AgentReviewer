# arxiv-search-kit

Offline ArXiv paper search over 928K CS papers. SPECTER2 embeddings + LanceDB vector index + BM25 hybrid retrieval.

**40ms per search on GPU. 99% precision@10. No API keys required. No rate limits.**

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

client = ArxivClient()  # downloads index on first run

results = client.search("attention mechanism transformers")
for paper in results:
    print(paper.title, paper.arxiv_id)
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

Find papers similar to a given paper using its stored SPECTER2 embedding. No keyword query needed.

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

## Async Support

All main methods have async variants:

```python
results = await client.async_search("transformers", max_results=10)
results = await client.async_batch_search(queries=[...], sort_by="importance")
related = await client.async_find_related("1706.03762")
await client.async_enrich(results)
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

## How It Works

1. **Index**: 928K papers embedded with [SPECTER2](https://huggingface.co/allenai/specter2), stored in [LanceDB](https://lancedb.github.io/lancedb/) (~4GB)
2. **Retrieval**: Hybrid search — dense (SPECTER2 cosine) + sparse (BM25) fused via Reciprocal Rank Fusion
3. **Re-ranking**: Personalized PageRank on a k-NN similarity graph built from candidate embeddings
4. **Enrichment**: Optional citation/venue data from [Semantic Scholar API](https://api.semanticscholar.org/api-docs/graph)
5. **Importance**: Blends relevance with citation count, venue prestige, and influential citation ratio

The index auto-downloads from [HuggingFace](https://huggingface.co/datasets/anonymousatom/arxiv-search-index) on first use.

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