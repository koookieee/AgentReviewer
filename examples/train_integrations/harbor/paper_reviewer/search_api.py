"""
arxiv-search-kit HTTP API server.

Runs on the training machine. Claude Code agents in E2B sandboxes
hit this API via ngrok to search for papers.

Exposes all arxiv-search-kit functionality: search, batch_search,
find_related, enrich, citations/references, and download.

Usage:
    python search_api.py [--port 4002]
"""

import argparse
import logging
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("search-api")

_client = None


def _get_client():
    global _client
    if _client is None:
        import torch
        from arxiv_search_kit import ArxivClient
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Initializing ArxivClient on {device} (first call downloads index from HF)...")
        _client = ArxivClient(device=device)
        log.info(f"ArxivClient ready on {device}.")
    return _client


def _paper_to_dict(paper):
    d = {
        "arxiv_id": getattr(paper, "arxiv_id", ""),
        "title": getattr(paper, "title", ""),
        "abstract": getattr(paper, "abstract", ""),
        "authors": getattr(paper, "author_names", []),
        "categories": getattr(paper, "categories", []),
        "primary_category": getattr(paper, "primary_category", ""),
        "published": str(getattr(paper, "published", "")),
        "year": getattr(paper, "year", None),
        "similarity_score": getattr(paper, "similarity_score", 0.0),
        "pdf_url": getattr(paper, "pdf_url", ""),
        "abs_url": getattr(paper, "abs_url", ""),
        "doi": getattr(paper, "doi", None),
        "journal_ref": getattr(paper, "journal_ref", None),
        "comment": getattr(paper, "comment", None),
    }
    # Enrichment fields (populated after enrich or sort_by="importance")
    if getattr(paper, "citation_count", None) is not None:
        d["citation_count"] = paper.citation_count
    if getattr(paper, "influential_citation_count", None) is not None:
        d["influential_citation_count"] = paper.influential_citation_count
    if getattr(paper, "venue", None):
        d["venue"] = paper.venue
    if getattr(paper, "tldr", None):
        d["tldr"] = paper.tldr
    if getattr(paper, "publication_types", None):
        d["publication_types"] = paper.publication_types
    return d


def _build_search_kwargs(body):
    """Extract common search kwargs from request body."""
    kwargs = {}
    for key in ("max_results", "categories", "year", "date_from", "date_to",
                "conference", "min_citations", "sort_by",
                "context_paper_id", "context_title", "context_abstract"):
        val = body.get(key)
        if val is not None:
            kwargs[key] = val
    return kwargs


async def handle_search(request: web.Request) -> web.Response:
    body = await request.json()
    query = body.get("query", "")
    kwargs = _build_search_kwargs(body)
    log.info(f"search: query={query!r} kwargs={kwargs}")

    client = _get_client()
    result = client.search(query, **kwargs)
    papers = [_paper_to_dict(p) for p in result.papers]
    return web.json_response({
        "papers": papers,
        "query": result.query,
        "total": len(papers),
    })


async def handle_batch_search(request: web.Request) -> web.Response:
    body = await request.json()
    queries = body.get("queries", [])
    kwargs = _build_search_kwargs(body)
    log.info(f"batch_search: {len(queries)} queries, kwargs={kwargs}")

    client = _get_client()
    result = client.batch_search(queries, **kwargs)
    papers = [_paper_to_dict(p) for p in result.papers]
    return web.json_response({
        "papers": papers,
        "total": len(papers),
    })


async def handle_find_related(request: web.Request) -> web.Response:
    body = await request.json()
    arxiv_id = body.get("arxiv_id", "")
    max_results = body.get("max_results", 10)
    categories = body.get("categories", None)
    log.info(f"find_related: arxiv_id={arxiv_id}")

    client = _get_client()
    kwargs = {"max_results": max_results}
    if categories:
        kwargs["categories"] = categories
    result = client.find_related(arxiv_id, **kwargs)
    papers = [_paper_to_dict(p) for p in result.papers]
    return web.json_response({"papers": papers, "total": len(papers)})


async def handle_enrich(request: web.Request) -> web.Response:
    """Enrich papers with citation data, venue, tldr from Semantic Scholar."""
    from arxiv_search_kit import SearchResult

    body = await request.json()
    arxiv_ids = body.get("arxiv_ids", [])
    fields = body.get("fields", None)
    log.info(f"enrich: {len(arxiv_ids)} papers")

    client = _get_client()
    # Look up each paper by ID
    papers = []
    for aid in arxiv_ids:
        p = client.get_paper(aid)
        if p is not None:
            papers.append(p)
    if papers:
        sr = SearchResult(papers=papers, query="enrich", total_candidates=len(papers), search_time_ms=0)
        enrich_kwargs = {}
        if fields:
            enrich_kwargs["fields"] = fields
        client.enrich(sr, **enrich_kwargs)
    return web.json_response({
        "papers": [_paper_to_dict(p) for p in papers],
        "total": len(papers),
    })


async def handle_citations(request: web.Request) -> web.Response:
    body = await request.json()
    arxiv_id = body.get("arxiv_id", "")
    limit = body.get("limit", 50)
    log.info(f"citations: arxiv_id={arxiv_id} limit={limit}")

    client = _get_client()
    citations = client.get_citations(arxiv_id, limit=limit)
    return web.json_response({"citations": citations, "total": len(citations)})


async def handle_references(request: web.Request) -> web.Response:
    body = await request.json()
    arxiv_id = body.get("arxiv_id", "")
    limit = body.get("limit", 50)
    log.info(f"references: arxiv_id={arxiv_id} limit={limit}")

    client = _get_client()
    references = client.get_references(arxiv_id, limit=limit)
    return web.json_response({"references": references, "total": len(references)})


async def handle_get_paper(request: web.Request) -> web.Response:
    body = await request.json()
    arxiv_id = body.get("arxiv_id", "")
    log.info(f"get_paper: arxiv_id={arxiv_id}")

    client = _get_client()
    paper = client.get_paper(arxiv_id)
    if paper is None:
        return web.json_response({"error": "Paper not found"}, status=404)
    return web.json_response({"paper": _paper_to_dict(paper)})


_extracted_cache: dict[str, str] = {}  # arxiv_id -> extract_dir path


def _extract_source(arxiv_id: str) -> tuple[str | None, str | None]:
    """Download + extract source for an arxiv paper. Returns (extract_dir, error)."""
    import os, tarfile, tempfile, gzip

    if arxiv_id in _extracted_cache:
        d = _extracted_cache[arxiv_id]
        if os.path.isdir(d):
            return d, None
        del _extracted_cache[arxiv_id]

    client = _get_client()
    tmpdir = tempfile.mkdtemp(prefix=f"arxiv_{arxiv_id}_")
    try:
        archive_path = client.download_source(arxiv_id, output_dir=tmpdir)
    except Exception as e:
        return None, f"Download failed: {e}"

    if archive_path is None or not os.path.exists(archive_path):
        return None, f"Source not available for {arxiv_id}"

    extract_dir = os.path.join(tmpdir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir, filter="data")
    except Exception:
        try:
            with gzip.open(archive_path, "rb") as gz:
                content = gz.read()
            with open(os.path.join(extract_dir, "main.tex"), "wb") as f:
                f.write(content)
        except Exception as e:
            return None, f"Extraction failed: {e}"

    _extracted_cache[arxiv_id] = extract_dir
    return extract_dir, None


async def handle_download_source(request: web.Request) -> web.Response:
    """Download + extract a paper's LaTeX source. Returns file listing (not content)."""
    import os

    body = await request.json()
    arxiv_id = body.get("arxiv_id", "")
    log.info(f"download_source: arxiv_id={arxiv_id}")

    extract_dir, err = _extract_source(arxiv_id)
    if err:
        return web.json_response({"error": err}, status=404 if "not available" in err else 500)

    files = []
    for root, dirs, filenames in os.walk(extract_dir):
        for fname in filenames:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, extract_dir)
            size = os.path.getsize(fpath)
            files.append({"path": rel, "size_bytes": size})
    files.sort(key=lambda f: f["path"])

    return web.json_response({
        "arxiv_id": arxiv_id,
        "files": files,
        "total_files": len(files),
    })


async def handle_read_file(request: web.Request) -> web.Response:
    """Read a specific file from a previously downloaded paper source."""
    import os

    body = await request.json()
    arxiv_id = body.get("arxiv_id", "")
    file_path = body.get("file_path", "")
    max_chars = body.get("max_chars", 100000)
    log.info(f"read_file: arxiv_id={arxiv_id} file={file_path}")

    extract_dir, err = _extract_source(arxiv_id)
    if err:
        return web.json_response({"error": err}, status=404 if "not available" in err else 500)

    full_path = os.path.normpath(os.path.join(extract_dir, file_path))
    if not full_path.startswith(extract_dir):
        return web.json_response({"error": "Invalid path"}, status=400)
    if not os.path.isfile(full_path):
        return web.json_response({"error": f"File not found: {file_path}"}, status=404)

    try:
        with open(full_path, "r", errors="replace") as f:
            content = f.read(max_chars)
        truncated = os.path.getsize(full_path) > max_chars
    except Exception as e:
        return web.json_response({"error": f"Read failed: {e}"}, status=500)

    return web.json_response({
        "arxiv_id": arxiv_id,
        "file_path": file_path,
        "content": content,
        "truncated": truncated,
        "size_bytes": os.path.getsize(full_path),
    })


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


app = web.Application()
app.router.add_get("/health", handle_health)
app.router.add_post("/search", handle_search)
app.router.add_post("/batch_search", handle_batch_search)
app.router.add_post("/find_related", handle_find_related)
app.router.add_post("/enrich", handle_enrich)
app.router.add_post("/citations", handle_citations)
app.router.add_post("/references", handle_references)
app.router.add_post("/get_paper", handle_get_paper)
app.router.add_post("/download_source", handle_download_source)
app.router.add_post("/read_file", handle_read_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4002)
    args = parser.parse_args()
    log.info(f"Starting search API on :{args.port}")
    web.run_app(app, port=args.port, print=None)