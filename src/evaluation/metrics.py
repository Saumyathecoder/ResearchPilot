"""RAG-specific evaluation metrics.

Retrieval quality and generation quality are measured independently,
because a pipeline can fail at either stage without failing the other.
"""
from __future__ import annotations

from src.graders.generation_grader import grade_generation


def retrieval_precision_at_k(retrieved_chunks: list[dict], expected_doc_ids: list[str]) -> float:
    """Fraction of retrieved chunks whose source doc_id is in the expected set.

    If `expected_doc_ids` is empty (the question is genuinely out of scope),
    precision is defined as 1.0 when nothing was retrieved, 0.0 otherwise.
    """
    if not expected_doc_ids:
        return 1.0 if not retrieved_chunks else 0.0
    if not retrieved_chunks:
        return 0.0
    hits = sum(1 for c in retrieved_chunks if c.get("doc_id") in expected_doc_ids)
    return hits / len(retrieved_chunks)


def retrieval_recall_at_k(retrieved_chunks: list[dict], expected_doc_ids: list[str]) -> float:
    """Fraction of expected docs that appear at least once among retrieved chunks."""
    if not expected_doc_ids:
        return 1.0
    retrieved_doc_ids = {c.get("doc_id") for c in retrieved_chunks}
    hits = sum(1 for d in expected_doc_ids if d in retrieved_doc_ids)
    return hits / len(expected_doc_ids)


def faithfulness_and_relevance(question: str, context_chunks: list[dict], answer: str) -> dict:
    """LLM-as-judge faithfulness (groundedness) and answer-relevance scores.

    Returns a dict with boolean `grounded`, `addresses_question`, and the
    combined `passed` flag, reusing the same grader used at inference time
    so evaluation measures exactly what the running system checks.
    """
    verdict = grade_generation(question, context_chunks, answer)
    return {
        "grounded": verdict.grounded,
        "addresses_question": verdict.addresses_question,
        "passed": verdict.passed,
        "reason": verdict.reason,
    }
