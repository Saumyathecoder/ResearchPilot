"""Wires the agentic RAG graph together.

                                 ┌────────────┐
                     ┌──────────►  direct     ├──────────────┐
                     │           └────────────┘              │
        ┌────────┐   │           ┌────────────┐              │
   ──►  │ route  ├───┼──────────►│ out_of_scope├─────────────┤
        └────────┘   │           └────────────┘              │
                     │           ┌────────────┐               ▼
                     └──────────►│  retrieve  │           ┌────────┐
                                 └─────┬──────┘           │  END   │
                                       │                  └────────┘
                                       ▼                       ▲
                              ┌────────────────┐               │
                              │ grade_documents│               │
                              └───────┬────────┘               │
                     enough relevant? │  not enough & retries left
                        │             ▼                        │
                        │     ┌────────────────┐                │
                        │     │ transform_query├──► retrieve    │
                        │     └────────────────┘  (loop, bounded)
                        │     not enough & no retries left      │
                        │             ▼                         │
                        │     ┌────────────────┐                │
                        │     │   web_search   │                │
                        │     └───────┬────────┘                │
                        ▼             ▼                          │
                              ┌────────────────┐                 │
                              │    generate    │                 │
                              └───────┬────────┘                 │
                                      ▼                          │
                              ┌────────────────┐                 │
                              │grade_generation│                 │
                              └───────┬────────┘                 │
                     passed ──────────┼────────── failed & retries left
                        │             │                     │
                        ▼             │                     ▼
                    finalize ◄────────┘             regenerate (loop back
                        │                             to `generate`, bounded)
                        └─────────────────────────────────────────► END
                              failed & no retries left
                                      │
                                      ▼
                          finalize_with_caveat ──► END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    node_route,
    node_direct_answer,
    node_out_of_scope,
    node_retrieve,
    node_grade_documents,
    node_transform_query,
    node_web_search,
    node_generate,
    node_grade_generation,
    node_finalize,
    node_finalize_with_caveat,
)
from src.config import CONFIG


# --- conditional edge functions ---------------------------------------------


def _route_decision(state: AgentState) -> str:
    return {"retrieve": "retrieve", "direct": "direct", "out_of_scope": "out_of_scope"}[state["route"]]


def _after_grading(state: AgentState) -> str:
    ratio = (
        sum(1 for g in state["graded"] if g["relevant"]) / len(state["graded"])
        if state.get("graded")
        else 0.0
    )
    if ratio >= CONFIG.relevance_threshold:
        return "generate"
    if state.get('relevant_chunks') and (state.get('context_coverage') or {}).get('sufficient') is True:
        return 'generate'
    if state.get("rewrite_count", 0) < CONFIG.max_query_rewrites:
        return "transform_query"
    if CONFIG.enable_web_fallback and not state.get("uploaded_corpus"):
        return "web_search"
    return "generate"  # give up gracefully with whatever (possibly empty) relevant_chunks we have


def _after_generation_grading(state: AgentState) -> str:
    verdict = state["generation_verdict"]
    if verdict["grounded"] and verdict["addresses_question"]:
        return "finalize"
    if state.get("generation_retry_count", 0) < CONFIG.max_generation_retries:
        return "retry_generate"
    return "finalize_with_caveat"


def _after_generation(state: AgentState) -> str:
    # A deterministic lack-of-evidence refusal is not a generated factual answer.
    # Never send it through answer-quality retries or invent a passing verdict.
    return "finalize" if state.get("abstained") else "grade_generation"


def _bump_retry(state: AgentState) -> dict:
    return {"generation_retry_count": state.get("generation_retry_count", 0) + 1}


def build_graph(store=None):
    """Compile and return the agentic RAG graph."""
    graph = StateGraph(AgentState)

    graph.add_node("route", node_route)
    graph.add_node("direct", node_direct_answer)
    graph.add_node("out_of_scope", node_out_of_scope)
    graph.add_node("retrieve", node_retrieve if store is None else lambda state: node_retrieve(state, store))
    graph.add_node("grade_documents", node_grade_documents)
    graph.add_node("transform_query", node_transform_query)
    graph.add_node("web_search", node_web_search)
    graph.add_node("generate", node_generate)
    graph.add_node("bump_retry", _bump_retry)
    graph.add_node("grade_generation", node_grade_generation)
    graph.add_node("finalize", node_finalize)
    graph.add_node("finalize_with_caveat", node_finalize_with_caveat)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route", _route_decision, {"retrieve": "retrieve", "direct": "direct", "out_of_scope": "out_of_scope"}
    )
    graph.add_edge("direct", END)
    graph.add_edge("out_of_scope", END)

    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        _after_grading,
        {"generate": "generate", "transform_query": "transform_query", "web_search": "web_search"},
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("web_search", "generate")

    graph.add_conditional_edges("generate", _after_generation,
                                {"finalize": "finalize", "grade_generation": "grade_generation"})
    graph.add_conditional_edges(
        "grade_generation",
        _after_generation_grading,
        {
            "finalize": "finalize",
            "retry_generate": "bump_retry",
            "finalize_with_caveat": "finalize_with_caveat",
        },
    )
    graph.add_edge("bump_retry", "generate")
    graph.add_edge("finalize", END)
    graph.add_edge("finalize_with_caveat", END)

    return graph.compile()


def run_agent(question: str, *, store=None, uploaded_corpus: bool = False) -> AgentState:
    """Convenience entry point: run the full graph for a single question."""
    if uploaded_corpus and store is None:
        raise ValueError("An uploaded corpus requires an explicit session store.")
    app = build_graph(store)
    initial_state: AgentState = {
        "original_question": question,
        "uploaded_corpus": uploaded_corpus,
        "rewrite_count": 0,
        "generation_retry_count": 0,
        "abstained": False,
        'coverage_check_count': 0,
        "used_web_fallback": False,
        "trace": [],
    }
    final_state = app.invoke(initial_state)
    return final_state
