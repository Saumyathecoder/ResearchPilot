"""Routes an incoming question to one of: vectorstore retrieval, direct
answer (no retrieval needed), or out-of-scope refusal.

This is the first decision in the agent graph and prevents the classic
naive-RAG bug of always retrieving, even for questions the corpus can't
possibly answer (e.g. "what's 2+2" or "who are you").
"""
from __future__ import annotations

from src.llm_client import complete_json

_ROUTER_SYSTEM = """You are a routing component in a RAG system. The corpus \
covers: RAG fundamentals, corrective/self-RAG architectures, embeddings and \
vector search, and agentic system design / evaluation of RAG systems.

Given a user question, decide the route:
- "retrieve": the question is about the corpus topics above and needs \
  retrieved context to answer well.
- "direct": the question is generic small talk, arithmetic, or something \
  answerable without any domain-specific retrieval.
- "out_of_scope": the question is clearly unrelated to the corpus topics \
  and retrieval would not help (e.g. asking for a cooking recipe).

Respond with ONLY a JSON object: {"route": "retrieve" | "direct" | "out_of_scope", \
"reason": "<one short sentence>"}"""


def route_question(question: str) -> dict:
    """Return {"route": ..., "reason": ...}."""
    prompt = f'User question: "{question}"'
    result = complete_json(prompt, system=_ROUTER_SYSTEM, max_tokens=200)
    route = result.get("route", "retrieve")
    if route not in {"retrieve", "direct", "out_of_scope"}:
        route = "retrieve"
    return {"route": route, "reason": result.get("reason", "")}
