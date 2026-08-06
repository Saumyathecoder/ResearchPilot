"""Persistent vector store built on Chroma, with embeddings computed by our
own pluggable embedder (rather than Chroma's built-in embedding functions),
so swapping between TF-IDF and sentence-transformers is a one-line change.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import chromadb

from src.config import CONFIG
from src.ingestion.chunker import Chunk
from src.retrieval.embeddings import BaseEmbedder, get_embedder

logger = logging.getLogger(__name__)

_EMBEDDER_CACHE_NAME = "embedder.pkl"


class VectorStore:
    """Thin wrapper around a Chroma collection.

    The embedder is fit once at ingestion time and persisted alongside the
    Chroma collection so that queries at retrieval time use the exact same
    fitted vocabulary/projection (this matters for TF-IDF; it is a no-op for
    pretrained neural embedders).
    """

    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        self.persist_dir = Path(persist_dir or CONFIG.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or CONFIG.collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.embedder: BaseEmbedder | None = self._load_embedder()

    # --- embedder persistence -------------------------------------------------
    def _embedder_path(self) -> Path:
        return self.persist_dir / f"{self.collection_name}.{_EMBEDDER_CACHE_NAME}"

    def _load_embedder(self) -> BaseEmbedder | None:
        path = self._embedder_path()
        if path.exists():
            with open(path, "rb") as f:
                logger.info("Loaded cached embedder from %s", path)
                return pickle.load(f)
        return None

    def _save_embedder(self) -> None:
        with open(self._embedder_path(), "wb") as f:
            pickle.dump(self.embedder, f)

    # --- ingestion -------------------------------------------------------------
    def index_chunks(self, chunks: list[Chunk], rebuild: bool = False) -> None:
        """Embed and index a list of chunks. `rebuild=True` wipes any
        existing collection first (use when the corpus changes)."""
        if rebuild:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

        texts = [c.text for c in chunks]
        self.embedder = get_embedder()
        self.embedder.fit(texts)
        vectors = self.embedder.embed(texts)
        self._save_embedder()

        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[
                {"doc_id": c.doc_id, "source_path": c.source_path, "section": c.section}
                for c in chunks
            ],
        )
        logger.info("Indexed %d chunks into collection '%s'", len(chunks), self.collection_name)

    # --- retrieval ---------------------------------------------------------
    def similarity_search(self, query: str, k: int | None = None) -> list[dict]:
        """Return top-k chunks as dicts with text, metadata, and a
        similarity score in [0, 1] (higher is more similar)."""
        if self.embedder is None or not self.embedder.is_fitted:
            raise RuntimeError(
                "Vector store has no fitted embedder. Run ingestion (index_chunks) first."
            )
        k = k or CONFIG.top_k
        query_vector = self.embedder.embed([query])[0].tolist()
        results = self.collection.query(query_embeddings=[query_vector], n_results=k)

        hits = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            # Chroma with cosine space returns distance = 1 - cosine_similarity
            similarity = max(0.0, 1.0 - distance)
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "similarity": similarity,
                    **meta,
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()
