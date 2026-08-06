"""A deliberately naive single-pass RAG baseline: retrieve top-k, generate,
no grading, no rewriting, no fallback. This exists purely so the evaluation
harness can show a concrete, measured before/after comparison against the
agentic pipeline, rather than asserting the agentic version is better.
"""
from __future__ import annotations

from src.config import CONFIG
from src.generation.generator import generate_answer
from src.retrieval.vectorstore import VectorStore

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def naive_rag_answer(question: str) -> dict:
    store = _get_store()
    retrieved = store.similarity_search(question, k=CONFIG.top_k)
    answer = generate_answer(question, retrieved)
    return {"retrieved": retrieved, "answer": answer}
