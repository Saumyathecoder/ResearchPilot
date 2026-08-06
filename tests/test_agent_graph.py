"""Tests for the agent graph's control flow.

LLM calls are monkeypatched so these tests run fully offline and fast,
verifying the *wiring* of the graph (routing, retry loops, fallback
triggers) rather than actual model quality. Retrieval itself uses the real
TF-IDF vector store since that requires no network access.
"""
import shutil

import pytest

import src.agent.nodes as nodes
from src.agent import graph as graph_module
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents
from src.retrieval.vectorstore import VectorStore


@pytest.fixture
def indexed_store(tmp_path):
    docs = load_documents("data/sample_docs")
    chunks = chunk_documents(docs)
    store = VectorStore(persist_dir=str(tmp_path / "chroma_graph_test"), collection_name="graph_test")
    store.index_chunks(chunks, rebuild=True)
    nodes._vector_store = store  # inject into the module-level singleton
    yield store
    nodes._vector_store = None
    shutil.rmtree(tmp_path, ignore_errors=True)


def _always_relevant_grader(question, chunk):
    from src.graders.relevance_grader import GradedChunk

    return GradedChunk(
        chunk_id=chunk["chunk_id"], text=chunk["text"], metadata=chunk, relevant=True, reason="mock"
    )


def _always_irrelevant_grader(question, chunk):
    from src.graders.relevance_grader import GradedChunk

    return GradedChunk(
        chunk_id=chunk["chunk_id"], text=chunk["text"], metadata=chunk, relevant=False, reason="mock"
    )


def test_happy_path_retrieves_and_finalizes(monkeypatch, indexed_store):
    monkeypatch.setattr(
        "src.agent.nodes.route_question", lambda q: {"route": "retrieve", "reason": "mock"}
    )
    monkeypatch.setattr("src.graders.relevance_grader.grade_chunk", _always_relevant_grader)
    monkeypatch.setattr("src.agent.nodes.generate_answer", lambda q, chunks: "A mocked grounded answer.")

    from src.graders.generation_grader import GenerationVerdict

    monkeypatch.setattr(
        "src.agent.nodes.grade_generation",
        lambda q, chunks, answer: GenerationVerdict(
            grounded=True, addresses_question=True, unsupported_claims=[], reason="mock"
        ),
    )

    final_state = graph_module.run_agent("What is dense vs sparse retrieval?")

    assert final_state["route"] == "retrieve"
    assert final_state["answer"] == "A mocked grounded answer."
    assert any("finalize" in step for step in final_state["trace"])
    assert final_state["rewrite_count"] == 0


def test_out_of_scope_question_skips_retrieval(monkeypatch, indexed_store):
    monkeypatch.setattr(
        "src.agent.nodes.route_question", lambda q: {"route": "out_of_scope", "reason": "mock"}
    )
    final_state = graph_module.run_agent("What's the best tiramisu recipe?")

    assert final_state["route"] == "out_of_scope"
    assert "outside the scope" in final_state["answer"]
    assert "retrieved" not in final_state  # retrieval node never ran


def test_low_relevance_triggers_query_rewrite_then_succeeds(monkeypatch, indexed_store):
    monkeypatch.setattr(
        "src.agent.nodes.route_question", lambda q: {"route": "retrieve", "reason": "mock"}
    )
    # first grading pass: irrelevant (forces rewrite), then relevant on retry
    call_count = {"n": 0}

    def grader(question, chunk):
        call_count["n"] += 1
        if call_count["n"] <= 4:  # first retrieval batch (top_k=4) graded irrelevant
            return _always_irrelevant_grader(question, chunk)
        return _always_relevant_grader(question, chunk)

    monkeypatch.setattr("src.graders.relevance_grader.grade_chunk", grader)
    monkeypatch.setattr("src.agent.nodes.rewrite_query", lambda orig, prev: "rewritten query")
    monkeypatch.setattr("src.agent.nodes.generate_answer", lambda q, chunks: "Answer after rewrite.")

    from src.graders.generation_grader import GenerationVerdict

    monkeypatch.setattr(
        "src.agent.nodes.grade_generation",
        lambda q, chunks, answer: GenerationVerdict(
            grounded=True, addresses_question=True, unsupported_claims=[], reason="mock"
        ),
    )

    final_state = graph_module.run_agent("obscure question unlikely to match well")

    assert final_state["rewrite_count"] == 1
    assert any("transform_query" in step for step in final_state["trace"])
    assert final_state["answer"] == "Answer after rewrite."


def test_hallucinated_generation_triggers_retry_then_caveat(monkeypatch, indexed_store):
    monkeypatch.setattr(
        "src.agent.nodes.route_question", lambda q: {"route": "retrieve", "reason": "mock"}
    )
    monkeypatch.setattr("src.graders.relevance_grader.grade_chunk", _always_relevant_grader)
    monkeypatch.setattr("src.agent.nodes.generate_answer", lambda q, chunks: "A possibly ungrounded answer.")

    from src.graders.generation_grader import GenerationVerdict
    from src.config import CONFIG

    # always fail groundedness -> should retry up to max_generation_retries then caveat-finalize
    monkeypatch.setattr(
        "src.agent.nodes.grade_generation",
        lambda q, chunks, answer: GenerationVerdict(
            grounded=False, addresses_question=True, unsupported_claims=["x"], reason="mock"
        ),
    )

    final_state = graph_module.run_agent("What is dense vs sparse retrieval?")

    assert final_state["generation_retry_count"] == CONFIG.max_generation_retries
    assert "[Note:" in final_state["answer"]
    assert any("finalize_with_caveat" in step for step in final_state["trace"])
