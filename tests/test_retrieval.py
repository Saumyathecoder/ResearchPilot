import shutil

import pytest

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents
from src.retrieval.embeddings import TfidfEmbedder
from src.retrieval.vectorstore import VectorStore


def test_tfidf_embedder_produces_normalized_vectors():
    embedder = TfidfEmbedder(n_components=8)
    texts = [
        "Retrieval augmented generation combines retrieval and generation.",
        "Vector databases index embeddings for nearest neighbor search.",
        "Chunking strategy affects retrieval quality significantly.",
    ]
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    assert vectors.shape[0] == len(texts)
    norms = (vectors**2).sum(axis=1) ** 0.5
    assert all(abs(n - 1.0) < 1e-3 for n in norms)


def test_tfidf_embedder_raises_if_not_fitted():
    embedder = TfidfEmbedder()
    with pytest.raises(RuntimeError):
        embedder.embed(["some text"])


@pytest.fixture
def populated_store(tmp_path):
    docs = load_documents("data/sample_docs")
    chunks = chunk_documents(docs)
    store = VectorStore(persist_dir=str(tmp_path / "chroma_test"), collection_name="test_collection")
    store.index_chunks(chunks, rebuild=True)
    yield store
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_vectorstore_indexes_and_retrieves(populated_store):
    assert populated_store.count() > 0

    hits = populated_store.similarity_search("What is dense vs sparse retrieval?", k=3)
    assert len(hits) == 3
    assert all("text" in h and "similarity" in h for h in hits)
    # the embeddings doc should be among top hits for this query
    doc_ids = {h["doc_id"] for h in hits}
    assert "embeddings_and_vector_search" in doc_ids


def test_vectorstore_query_before_index_raises(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "empty_store"), collection_name="empty")
    with pytest.raises(RuntimeError):
        store.similarity_search("anything")
