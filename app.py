"""Streamlit demo UI for the agentic RAG system.

Run locally with:
    streamlit run app.py

Deploys as-is to Streamlit Community Cloud (share.streamlit.io) - the vector
store is built automatically on first load, so no manual ingestion step is
needed after deployment.
"""
from __future__ import annotations

import streamlit as st

from src.agent.graph import run_agent
from src.config import CONFIG
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents
from src.retrieval.vectorstore import VectorStore

st.set_page_config(page_title="Agentic Self-Correcting RAG", page_icon="🔎", layout="wide")

EXAMPLE_QUESTIONS = [
    "What three actions can Corrective RAG take based on retrieval confidence?",
    "What is the difference between groundedness and answer relevance?",
    "Why does chunk size affect retrieval quality?",
    "What's the best recipe for tiramisu?",  # deliberately out-of-scope, to show routing
]


@st.cache_resource(show_spinner=False)
def get_ready_vector_store() -> VectorStore:
    """Build the vector store on first load if it's empty. Cached across
    reruns/sessions for the lifetime of the deployed app process."""
    store = VectorStore()
    if store.count() == 0:
        documents = load_documents("data/sample_docs")
        chunks = chunk_documents(documents)
        store.index_chunks(chunks, rebuild=True)
    return store


st.title("🔎 Agentic Self-Correcting RAG")
st.caption(
    "route → retrieve → grade → (rewrite / web-fallback) → generate → self-critique → finalize — "
    "an explicit LangGraph state machine, not a single retrieve-then-generate pass."
)

key_present = (CONFIG.llm_provider == "groq" and CONFIG.groq_api_key) or (
    CONFIG.llm_provider == "anthropic" and CONFIG.anthropic_api_key
)
if not key_present:
    st.warning(
        f"No API key configured for provider `{CONFIG.llm_provider}`. "
        "Set it in `.env` (local) or in your Streamlit Cloud app secrets (deployed) "
        "before asking a question.",
        icon="⚠️",
    )

with st.sidebar:
    st.header("How this works")
    st.markdown(
        "- **Route**: is this question in-scope for the corpus, general knowledge, "
        "or out of scope?\n"
        "- **Retrieve + grade**: each retrieved chunk is graded for relevance by "
        "an LLM call, not just trusted on embedding similarity alone.\n"
        "- **Self-correct**: low relevance triggers query rewriting, then falls "
        "back to web search if rewrites don't help.\n"
        "- **Generate + self-critique**: the answer is checked for groundedness "
        "and relevance *after* generation, and regenerated if it fails either "
        "check.\n"
    )
    st.divider()
    st.header("Corpus")
    st.write(
        "Grounded in a small knowledge base covering RAG fundamentals, "
        "Corrective/Self-RAG, embeddings & vector search, and agentic "
        "system evaluation."
    )
    st.caption("[View source on GitHub](.)")

with st.spinner("Setting up knowledge base (first load only)..."):
    get_ready_vector_store()

st.subheader("Try it")
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, ex in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input(
    "Ask a question",
    key="question",
    placeholder="e.g. What three actions can Corrective RAG take based on retrieval confidence?",
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Running agent graph..."):
        state = run_agent(question)

    st.subheader("Answer")
    st.write(state.get("answer", "(no answer produced)"))

    verdict = state.get("generation_verdict")
    if verdict:
        c1, c2, c3 = st.columns(3)
        c1.metric("Route", state.get("route", "n/a"))
        c2.metric("Grounded", "✅" if verdict["grounded"] else "❌")
        c3.metric("Addresses question", "✅" if verdict["addresses_question"] else "❌")

    with st.expander("Agent trace (every decision the graph made, in order)", expanded=True):
        for line in state.get("trace", []):
            st.text(line)

    chunks = state.get("relevant_chunks", [])
    if chunks:
        with st.expander(f"Context actually used ({len(chunks)} chunks)"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**[{i}] {c.get('source_path', 'web')}** — _{c.get('section', 'n/a')}_")
                st.text(c["text"][:600] + ("..." if len(c["text"]) > 600 else ""))
