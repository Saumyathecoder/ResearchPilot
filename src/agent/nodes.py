"""Node implementations for the self-correcting agentic RAG graph.

Each function takes the current AgentState and returns a partial state
update (LangGraph merges dict returns into the running state). Keeping each
node as a small, pure-ish function makes the control flow in graph.py easy
to read and each step easy to unit test in isolation.
"""
from __future__ import annotations

import logging

from src.config import CONFIG
from src.generation.generator import generate_answer
from src.graders.generation_grader import grade_generation
from src.graders.query_rewriter import rewrite_query
from src.graders.relevance_grader import grade_chunks, relevance_ratio
from src.graders.router import route_question
from src.graders.coverage_grader import check_context_coverage
from src.llm_client import LLMError
from src.retrieval.vectorstore import VectorStore
from src.tools.web_search import web_search
from src.agent.state import AgentState

logger = logging.getLogger(__name__)

_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _log(state: AgentState, message: str) -> list[str]:
    trace = list(state.get("trace", []))
    trace.append(message)
    logger.info(message)
    return trace


# --- nodes -------------------------------------------------------------------


def node_route(state: AgentState) -> dict:
    result = ({"route": "retrieve", "reason": "Search the selected uploaded documents, regardless of topic."}
              if state.get("uploaded_corpus") else route_question(state["original_question"]))
    trace = _log(state, f"[route] -> {result['route']} ({result['reason']})")
    return {"route": result["route"], "route_reason": result["reason"], "trace": trace,
            "query": state["original_question"]}


def node_direct_answer(state: AgentState) -> dict:
    from src.llm_client import complete

    answer = complete(
        state["original_question"],
        system="Answer directly and concisely. No need to cite sources for this general question.",
    )
    trace = _log(state, "[direct_answer] answered without retrieval")
    return {"answer": answer, "trace": trace}


def node_out_of_scope(state: AgentState) -> dict:
    answer = (
        "This question is outside the scope of the knowledge base this assistant "
        "is grounded in (RAG architectures, embeddings/vector search, and agentic "
        "system evaluation), so I can't answer it reliably."
    )
    trace = _log(state, "[out_of_scope] declined to answer")
    return {"answer": answer, "trace": trace}


def node_retrieve(state: AgentState, store=None) -> dict:
    store = store if store is not None else get_vector_store()
    hits = store.similarity_search(state["query"], k=CONFIG.top_k)
    trace = _log(state, f"[retrieve] query='{state['query']}' -> {len(hits)} chunks")
    return {"retrieved": hits, "trace": trace}


def node_grade_documents(state: AgentState) -> dict:
    graded = grade_chunks(state["original_question"], state["retrieved"])
    ratio = relevance_ratio(graded)
    relevant_chunks = [
        {"chunk_id": g.chunk_id, "text": g.text, **g.metadata} for g in graded if g.relevant
    ]
    trace = _log(
        state,
        f"[grade_documents] {sum(g.relevant for g in graded)}/{len(graded)} relevant "
        f"(ratio={ratio:.2f}, threshold={CONFIG.relevance_threshold})",
    )
    coverage = None
    checks = state.get('coverage_check_count', 0)
    if CONFIG.enable_context_coverage and relevant_chunks and ratio < CONFIG.relevance_threshold:
        checks += 1
        try:
            coverage = check_context_coverage(state['original_question'], relevant_chunks)
            trace = _log({'trace': trace}, f"[context_coverage] sufficient={coverage['sufficient']}; missing_aspects={len(coverage['missing_aspects'])}")
        except LLMError:
            # A failed check must never grant permission to skip retrieval recovery.
            trace = _log({'trace': trace}, '[context_coverage] unavailable; retaining relevance-ratio policy')
    return {
        "graded": [g.__dict__ for g in graded],
        "relevant_chunks": relevant_chunks,
        "trace": trace,
        'context_coverage': coverage,
        'coverage_check_count': checks,
    }


def node_transform_query(state: AgentState) -> dict:
    new_query = rewrite_query(state["original_question"], state["query"])
    count = state.get("rewrite_count", 0) + 1
    trace = _log(state, f"[transform_query] rewrite #{count}: '{new_query}'")
    return {"query": new_query, "rewrite_count": count, "trace": trace}


def node_web_search(state: AgentState) -> dict:
    results = web_search(state["original_question"])
    trace = _log(state, f"[web_search] fallback returned {len(results)} results")
    combined = state.get("relevant_chunks", []) + results
    return {"relevant_chunks": combined, "used_web_fallback": True, "trace": trace}


def node_generate(state: AgentState) -> dict:
    chunks = state.get("relevant_chunks", [])
    if not chunks:
        answer = "I don't have relevant context to answer this question reliably."
        if state.get("uploaded_corpus"):
            answer += " No relevant passages were found in your uploaded documents. Web search is disabled for uploads."
        elif state.get("used_web_fallback"):
            answer += " Local retrieval and web search did not provide usable context."
        trace = _log(state, "[abstain] no relevant context; skipping generation and self-review")
        return {"generation": answer, "abstained": True, "trace": trace}
    answer = generate_answer(state["original_question"], chunks)
    trace = _log(state, "[generate] produced draft answer")
    return {"generation": answer, "abstained": False, "trace": trace}


def node_grade_generation(state: AgentState) -> dict:
    verdict = grade_generation(
        state["original_question"], state.get("relevant_chunks", []), state["generation"]
    )
    trace = _log(
        state,
        f"[grade_generation] grounded={verdict.grounded} "
        f"addresses_question={verdict.addresses_question}",
    )
    return {"generation_verdict": verdict.__dict__, "trace": trace}


def node_finalize(state: AgentState) -> dict:
    trace = _log(state, "[finalize] returning generation as final answer")
    return {"answer": state["generation"], "trace": trace}


def node_finalize_with_caveat(state: AgentState) -> dict:
    caveat = (
        "\n\n[Note: after self-review, this answer may not be fully grounded in "
        "the retrieved sources or may not fully address your question. Treat it "
        "with appropriate caution.]"
    )
    trace = _log(state, "[finalize_with_caveat] max retries reached, returning best-effort answer")
    return {"answer": state["generation"] + caveat, "trace": trace}
