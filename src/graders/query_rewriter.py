"""Rewrites a query that failed to retrieve relevant chunks.

Vocabulary mismatch between how a user phrases a question and how the
answer is phrased in the corpus is one of the most common causes of
retrieval failure. Rewriting the query (e.g. making it more specific, or
restating it using terminology likely to appear in the corpus) and
re-retrieving is cheap and often fixes this without needing a fallback to
external search.
"""
from __future__ import annotations

from src.llm_client import complete

_REWRITE_SYSTEM = """You rewrite search queries to improve retrieval from a \
technical knowledge base about RAG systems, embeddings, and agentic AI. \
The previous query did not retrieve sufficiently relevant passages.

Rewrite the query to be clearer, more specific, and more likely to match \
how this topic would be described in technical documentation. Respond with \
ONLY the rewritten query text, nothing else."""


def rewrite_query(original_question: str, previous_query: str) -> str:
    prompt = (
        f'Original user question: "{original_question}"\n'
        f'Previous search query (retrieved poor results): "{previous_query}"'
    )
    rewritten = complete(prompt, system=_REWRITE_SYSTEM, max_tokens=100).strip().strip('"')
    return rewritten or previous_query
