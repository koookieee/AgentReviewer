# search-papers

Search academic literature across multiple providers: Semantic Scholar, OpenAlex, arXiv, OpenReview, and CrossRef.

## When to use

Use this skill whenever you need to:
- Find related work for a research paper
- Verify novelty claims against existing literature
- Check if baselines are actually SOTA
- Look up paper metadata (citations, authors, venues)
- Download PDFs for deeper reading
- Get BibTeX entries

## How to use

Run `curl` commands from the terminal to query academic APIs. See `reference.md` in this directory for the full API reference with examples.

### Quick examples

**Search by keyword (Semantic Scholar):**
```bash
curl -s -H "x-api-key: $S2_API_KEY" \
  "https://api.semanticscholar.org/graph/v1/paper/search?query=your+search+terms&limit=10&fields=title,authors,year,abstract,citationCount,externalIds"
```

**Search by keyword (OpenAlex):**
```bash
curl -s "https://api.openalex.org/works?search=your+search+terms&sort=cited_by_count:desc&per-page=10&select=id,title,publication_year,cited_by_count,doi&api_key=$OPENALEX_API_KEY"
```

**Look up a paper by arXiv ID:**
```bash
curl -s -H "x-api-key: $S2_API_KEY" \
  "https://api.semanticscholar.org/graph/v1/paper/ArXiv:2301.00001?fields=title,abstract,authors,citationCount,externalIds"
```

**Download an arXiv PDF:**
```bash
mkdir -p literature
curl -s -L -o literature/paper.pdf "https://arxiv.org/pdf/2301.00001"
```

**Get BibTeX for a DOI:**
```bash
curl -s -L -H "Accept: application/x-bibtex" "https://dx.doi.org/10.48550/arXiv.2301.00001"
```

## Available providers

| Provider | Best for | Auth |
|----------|----------|------|
| Semantic Scholar | Keyword search, citations graph, paper lookup | `$S2_API_KEY` header |
| OpenAlex | OA PDF links, citation counts, filtering | `$OPENALEX_API_KEY` param |
| arXiv | Direct PDF download, HTML reading | None |
| OpenReview | ICLR/NeurIPS/ICML papers, reviews, decisions | None |
| CrossRef | BibTeX from any DOI | None |

## Rate limits

- **Semantic Scholar:** 1 req/s with API key. Back off on 429.
- **OpenAlex:** 100 req/s. Daily budget $1 (search costs $0.001 each).
- **arXiv:** Wait 3s between downloads. No bulk scraping.
- **OpenReview:** ~5 concurrent requests OK.

See `reference.md` for complete API documentation with all endpoints, parameters, and response formats.
