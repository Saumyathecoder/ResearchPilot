"""Web search fallback, used when the vector store has no relevant chunks
for a query (CRAG-style fallback to external knowledge).

Uses the `ddgs` (DuckDuckGo Search) package, which requires no API key.
Degrades gracefully if the package is missing or the network call fails,
since this is a fallback path and should never crash the whole pipeline.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Return a list of {"text": ..., "source_path": ..., "section": ...}
    dicts shaped like retrieved chunks, so downstream code (generation,
    grading) can treat them identically to vector-store hits."""
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs package not installed; web fallback unavailable. `pip install ddgs`.")
        return []

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001 - fallback path must never crash the pipeline
        logger.warning("Web search failed (%s); continuing without web results.", exc)
        return []

    results = []
    for r in raw_results:
        results.append(
            {
                "chunk_id": f"web::{hash(r.get('href', r.get('title', '')))}",
                "text": f"{r.get('title', '')}\n{r.get('body', '')}",
                "source_path": r.get("href", "web"),
                "section": "web search result",
                "similarity": None,
            }
        )
    return results
