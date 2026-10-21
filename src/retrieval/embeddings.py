"""Pluggable embedding backends behind one common interface.

Two backends are supported:

* "tfidf" - scikit-learn TF-IDF vectorizer. Pure local, no downloads, works
  immediately offline. Lower semantic quality (lexical, not neural) but a
  perfectly good default for demos, CI, and getting the pipeline running
  with zero setup friction.
* "st"    - sentence-transformers dense embeddings. Much better semantic
  retrieval quality. Requires internet access on first run to download the
  model weights from Hugging Face; cached locally afterward.

Swap backends via `RAG_EMBEDDING_BACKEND` in `.env` (see config.py).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from src.config import CONFIG

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Fit the embedder on a corpus (no-op for pretrained neural models)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 matrix of embeddings."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        ...


class TfidfEmbedder(BaseEmbedder):
    """Local, dependency-light embedder using TF-IDF + SVD dimensionality
    reduction (so vectors behave like dense embeddings for cosine search)."""

    def __init__(self, n_components: int = 256):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), stop_words="english"
        )
        self.n_components = n_components
        self.svd = None
        self._fitted = False
        self._TruncatedSVD = TruncatedSVD

    def fit(self, texts: list[str]) -> None:
        sparse = self.vectorizer.fit_transform(texts)
        n_components = min(self.n_components, max(2, sparse.shape[1] - 1), sparse.shape[0] - 1)
        n_components = max(n_components, 2)
        self.svd = None
        if min(sparse.shape) > 2:
            self.svd = self._TruncatedSVD(n_components=n_components, random_state=42)
            self.svd.fit(sparse)
        self._fitted = True
        logger.info("TfidfEmbedder fitted on %d docs, %d SVD components", len(texts), n_components)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed().")
        sparse = self.vectorizer.transform(texts)
        dense = self.svd.transform(sparse) if self.svd is not None else sparse.toarray()
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense / norms).astype("float32")

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense neural embedder. Requires the `sentence-transformers` package
    and internet access to download weights on first use."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name or CONFIG.st_model_name)

    def fit(self, texts: list[str]) -> None:
        # Pretrained model: nothing to fit.
        return

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype="float32")

    @property
    def is_fitted(self) -> bool:
        return True


def get_embedder(backend: str | None = None) -> BaseEmbedder:
    backend = (backend or CONFIG.embedding_backend).lower()
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend in {"st", "sentence-transformers", "sentence_transformers"}:
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend!r}. Use 'tfidf' or 'st'.")
