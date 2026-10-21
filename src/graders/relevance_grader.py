"""Grades each retrieved chunk for relevance to the query (CRAG-style).

Rather than trusting embedding similarity alone (which can be topically
close but not actually answer the question), a second LLM pass explicitly
judges relevance. Irrelevant chunks are filtered out before generation, and
the aggregate relevance signal drives the agent's next action (proceed,
rewrite the query, or fall back to web search).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.llm_client import complete_json, validate_fields

_GRADER_SYSTEM = """You are a strict retrieval grader. Given a user question \
and a single retrieved passage, decide if the passage contains information \
that is directly relevant and useful for answering the question - not \
merely on the same general topic.

Respond with ONLY a JSON object: {"relevant": true|false, "reason": "<one short sentence>"}"""


@dataclass
class GradedChunk:
    chunk_id: str
    text: str
    metadata: dict
    relevant: bool
    reason: str


def grade_chunk(question: str, chunk: dict) -> GradedChunk:
    prompt = f'Question: "{question}"\n\nRetrieved passage:\n"""{chunk["text"]}"""'
    result = complete_json(prompt, system=_GRADER_SYSTEM, max_tokens=150,
                           validator=lambda r: validate_fields(r, {'relevant': bool, 'reason': str}))
    return GradedChunk(
        chunk_id=chunk["chunk_id"],
        text=chunk["text"],
        metadata={k: v for k, v in chunk.items() if k not in {"chunk_id", "text"}},
        relevant=result['relevant'],
        reason=result['reason'],
    )


def grade_chunks(question: str, chunks: list[dict]) -> list[GradedChunk]:
    """Grade every retrieved chunk. Sequential and simple by design; swap in
    concurrent.futures.ThreadPoolExecutor here if latency becomes an issue
    with a larger top_k."""
    return [grade_chunk(question, chunk) for chunk in chunks]


def relevance_ratio(graded: list[GradedChunk]) -> float:
    if not graded:
        return 0.0
    return sum(1 for g in graded if g.relevant) / len(graded)
