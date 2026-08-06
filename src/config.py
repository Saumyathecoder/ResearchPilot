"""Central configuration for the agentic RAG pipeline.

All tunables live here so behavior can be changed from one place (or from
environment variables) without hunting through the codebase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _load_streamlit_secrets_into_env() -> None:
    """When running under Streamlit Community Cloud, secrets are configured
    in the dashboard (not a local .env file) and exposed via st.secrets.
    Mirror them into os.environ so the rest of this module (and the whole
    codebase, which reads via os.getenv) doesn't need to know whether it's
    running locally or deployed. No-op outside a Streamlit context."""
    try:
        import streamlit as st

        for key, value in st.secrets.items():
            os.environ.setdefault(key, str(value))
    except Exception:
        pass  # not running under Streamlit, or no secrets configured - fine.


_load_streamlit_secrets_into_env()


@dataclass
class Config:
    # --- LLM ---
    # "anthropic" (paid) or "groq" (free tier, OpenAI-compatible API).
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "groq")
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    llm_model: str = os.getenv(
        "RAG_LLM_MODEL",
        "llama-3.3-70b-versatile" if os.getenv("RAG_LLM_PROVIDER", "groq") == "groq" else "claude-sonnet-4-6",
    )
    llm_max_tokens: int = int(os.getenv("RAG_LLM_MAX_TOKENS", "1024"))

    # --- Embeddings ---
    # "tfidf"    -> pure-local, zero downloads, works offline immediately.
    # "st"       -> sentence-transformers dense embeddings, better quality,
    #               requires internet on first run to download the model.
    embedding_backend: str = os.getenv("RAG_EMBEDDING_BACKEND", "tfidf")
    st_model_name: str = os.getenv("RAG_ST_MODEL", "all-MiniLM-L6-v2")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

    # --- Retrieval ---
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    persist_dir: str = os.getenv("RAG_PERSIST_DIR", "./chroma_db")
    collection_name: str = os.getenv("RAG_COLLECTION", "agentic_rag_docs")

    # --- Agent control flow ---
    relevance_threshold: float = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.5"))
    max_query_rewrites: int = int(os.getenv("RAG_MAX_REWRITES", "2"))
    max_generation_retries: int = int(os.getenv("RAG_MAX_GEN_RETRIES", "2"))
    enable_web_fallback: bool = os.getenv("RAG_ENABLE_WEB_FALLBACK", "true").lower() == "true"

    # --- Misc ---
    verbose: bool = os.getenv("RAG_VERBOSE", "true").lower() == "true"


CONFIG = Config()
