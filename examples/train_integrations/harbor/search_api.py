"""
arxiv-search-kit HTTP API server.

Runs on the training machine. Claude Code agents in E2B sandboxes
hit this API via ngrok to search for papers.

Usage:
    python search_api.py [--port 4002]
"""

import argparse
import json
import logging
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("search-api")

# Lazy-init so the HF index download only happens once
_client = None


def _get_client():
    global _client
    if _client is None:
        from arxiv_search_kit import ArxivClient
        log.info("Initializing ArxivClient (first call downloads index from HF)...")
        _client = ArxivClient()
        log.info("ArxivClient ready.")
    return _client


def _paper_to_dict(paper):
    return {
        "arxiv_id": getattr(paper, "arxiv_id", ""),
        "title": getattr(paper, "title", ""),
        "abstract": getattr(paper, "abstract", ""),
        "authors": getattr(paper, "authors", []),
        "categories": getattr(paper, "categories", []),
        "published": str(getattr(paper, "published", "")),
        "similarity_score": getattr(paper, "similarity_score", 0.0),
    }


async def handle_search(request: web.Request) -> web.Response:
    body = await request.json()
    query = body.get("query", "")
    max_results = body.get("max_results", 10)
    categories = body.get("categories", None)
    context_paper = body.get("context_paper", None)

    log.info(f"search: query={query!r} max_results={max_results} categories={categories}")

    client = _get_client()
    kwargs = {"max_results": max_results}
    if categories:
        kwargs["categories"] = categories
    if context_paper:
        kwargs["context_paper"] = context_paper

    result = client.search(query, **kwargs)
    papers = [_paper_to_dict(p) for p in result.papers]
    return web.json_response({"papers": papers})


async def handle_find_related(request: web.Request) -> web.Response:
    body = await request.json()
    arxiv_id = body.get("arxiv_id", "")
    max_results = body.get("max_results", 10)

    log.info(f"find_related: arxiv_id={arxiv_id} max_results={max_results}")

    client = _get_client()
    result = client.find_related(arxiv_id, max_results=max_results)
    papers = [_paper_to_dict(p) for p in result.papers]
    return web.json_response({"papers": papers})


async def handle_batch_search(request: web.Request) -> web.Response:
    body = await request.json()
    queries = body.get("queries", [])
    max_results = body.get("max_results", 10)
    categories = body.get("categories", None)

    log.info(f"batch_search: {len(queries)} queries, max_results={max_results}")

    client = _get_client()
    kwargs = {"max_results": max_results}
    if categories:
        kwargs["categories"] = categories

    result = client.batch_search(queries, **kwargs)
    papers = [_paper_to_dict(p) for p in result.papers]
    return web.json_response({"papers": papers})


async def handle_get_paper(request: web.Request) -> web.Response:
    body = await request.json()
    arxiv_id = body.get("arxiv_id", "")

    log.info(f"get_paper: arxiv_id={arxiv_id}")

    client = _get_client()
    paper = client.get_paper(arxiv_id)
    if paper is None:
        return web.json_response({"error": "Paper not found"}, status=404)
    return web.json_response({"paper": _paper_to_dict(paper)})


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


app = web.Application()
app.router.add_get("/health", handle_health)
app.router.add_post("/search", handle_search)
app.router.add_post("/find_related", handle_find_related)
app.router.add_post("/batch_search", handle_batch_search)
app.router.add_post("/get_paper", handle_get_paper)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4002)
    args = parser.parse_args()
    log.info(f"Starting search API on :{args.port}")
    web.run_app(app, port=args.port, print=None)