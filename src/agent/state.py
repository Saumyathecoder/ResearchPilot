"""Shared state that flows through every node of the agent graph."""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    # input
    original_question: str
    uploaded_corpus: bool
    # mutable working query (may be rewritten mid-run)
    query: str
    # routing
    route: str
    route_reason: str
    # retrieval
    retrieved: list[dict]
    graded: list[dict]  # serialized GradedChunk-like dicts
    relevant_chunks: list[dict]
    context_coverage: dict | None
    coverage_check_count: int
    used_web_fallback: bool
    rewrite_count: int
    # generation
    generation: str
    abstained: bool
    generation_verdict: dict
    generation_retry_count: int
    # output
    answer: str
    # observability: ordered list of human-readable steps taken
    trace: list[str]
