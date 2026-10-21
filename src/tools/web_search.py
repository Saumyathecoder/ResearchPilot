"""Web search fallback, used when the vector store has no relevant chunks
for a query (CRAG-style fallback to external knowledge).

Uses the `ddgs` (DuckDuckGo Search) package, which requires no API key.
Degrades gracefully if the package is missing or the network call fails,
since this is a fallback path and should never crash the whole pipeline.
"""
from __future__ import annotations

import logging
import hashlib
from urllib.parse import urlsplit

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
        with DDGS(timeout=5) as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001 - fallback path must never crash the pipeline
        logger.warning("Web search failed (%s); continuing without web results.", type(exc).__name__)
        return []

    results = []
    seen = set()
    for r in raw_results:
        if not isinstance(r, dict):
            continue
        url, body, title = r.get('href'), r.get('body'), r.get('title', '')
        if not isinstance(url, str) or not isinstance(body, str) or not body.strip():
            continue
        url = url.strip()
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {'http', 'https'} and bool(parsed.hostname) and not parsed.username and not parsed.password
        except ValueError:
            valid = False
        if not valid or url in seen:
            continue
        seen.add(url)
        title = title.strip() if isinstance(title, str) else ''
        results.append(
            {
                "chunk_id": f"web::{hashlib.sha256(url.encode()).hexdigest()}",
                "text": f"{title}\n{body.strip()}".strip(),
                "source_path": url,
                "section": "web search result",
                "similarity": None,
            }
        )
    return results
