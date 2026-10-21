"""ResearchPilot's document workspace. Run with streamlit run app.py."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.agent.graph import run_agent
from src.config import CONFIG
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import DocumentError, load_documents
from src.ingestion.uploads import UploadCorpus
from src.llm_client import LLMError
from src.retrieval.vectorstore import VectorStore

st.set_page_config(page_title="ResearchPilot", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{(Path(__file__).parent / 'assets/app.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

EXAMPLE_QUESTIONS = [
    ("Corrective RAG decisions", "What three actions can Corrective RAG take based on retrieval confidence?"),
    ("Groundedness vs. relevance", "What is the difference between groundedness and answer relevance?"),
    ("Chunk size", "Why does chunk size affect retrieval quality?"),
    ("An unrelated question", "What's the best recipe for tiramisu?"),
]


@st.cache_resource(show_spinner=False)
def get_ready_vector_store() -> VectorStore:
    """Only the public sample store is shared across sessions."""
    store = VectorStore()
    if store.count() == 0:
        store.index_chunks(chunk_documents(load_documents("data/sample_docs")), rebuild=True)
    return store


with st.sidebar:
    st.markdown('<div class="workspace-brand">ResearchPilot</div><div class="workspace-label">Document workspace</div>', unsafe_allow_html=True)
    st.subheader("Documents")
    corpus_mode = st.radio("Knowledge base", ["Sample demo", "My uploads"])
    if corpus_mode == "My uploads":
        files = st.file_uploader("PDF, Markdown, or UTF-8 text", type=["pdf", "md", "txt"],
                                 accept_multiple_files=True, max_upload_size=10)
        st.caption("Up to 10 files · 10 MB each · 30 MB total")
        if st.button("Index selected documents", use_container_width=True):
            try:
                with st.spinner("Preparing your documents…"):
                    corpus = UploadCorpus([(file.name, file.getvalue()) for file in files])
                st.session_state["upload_corpus"] = corpus
                st.success(f"Ready: {corpus.file_count} files, {corpus.count()} passages.")
            except DocumentError as exc:
                st.error(str(exc))
                if st.session_state.get("upload_corpus"):
                    st.info("The previous corpus remains active; see Active files below.")
            except Exception:
                st.error("Could not build the document index. Check the embedding backend and try again.")
        corpus = st.session_state.get("upload_corpus")
        if corpus:
            st.divider()
            st.caption("Active files")
            for name in corpus.filenames:
                st.text(name)
            for warning in corpus.warnings:
                st.warning(warning)
            if st.button("Clear uploaded corpus", use_container_width=True):
                st.session_state.pop("upload_corpus", None)
                st.rerun()
        else:
            st.info("Index documents before asking a question.")
        with st.expander("Upload limits & privacy"):
            st.caption("Files and indexes stay in this session's memory. Relevant passages and questions are sent to your configured AI provider. Web search is disabled for uploads.")
            st.caption("Text-based PDFs only; no OCR. Up to 200 pages per PDF. Refreshing or restarting may discard uploads.")
    else:
        st.caption("A small reference collection on retrieval, embeddings and agentic systems.")

    st.divider()
    with st.expander("How answers are checked"):
        st.caption("Passages are checked for relevance. If evidence is incomplete, the agent can rewrite the query or search the web when permitted. Answers are reviewed for grounding and relevance; missing evidence produces an explicit refusal.")

st.caption("WORKSPACE / " + ("SAMPLE COLLECTION" if corpus_mode == "Sample demo" else "YOUR DOCUMENTS"))
st.title("Research workspace")
st.caption("Ask a question. Read the evidence behind the answer.")

key_present = (CONFIG.llm_provider == "groq" and CONFIG.groq_api_key) or (
    CONFIG.llm_provider == "anthropic" and CONFIG.anthropic_api_key)
if not key_present:
    st.warning(f"No API key configured for {CONFIG.llm_provider}. Set it in your local environment or app secrets before asking a question.")

if corpus_mode == "Sample demo":
    with st.spinner("Opening sample collection…"):
        get_ready_vector_store()

if corpus_mode == "Sample demo":
    with st.expander("Example questions"):
        columns = st.columns(2)
        for index, (label, example) in enumerate(EXAMPLE_QUESTIONS):
            if columns[index % 2].button(label, use_container_width=True):
                st.session_state["question"] = example

question = st.text_input("Ask a question", key="question", placeholder="What would you like to find out?")
ready = corpus_mode == "Sample demo" or bool(st.session_state.get("upload_corpus"))
if st.button("Ask", type="primary", disabled=not ready) and question.strip():
    with st.spinner("Reading sources and checking the answer…"):
        try:
            if corpus_mode == "My uploads":
                state = run_agent(question, store=st.session_state["upload_corpus"], uploaded_corpus=True)
            else:
                state = run_agent(question)
        except LLMError as exc:
            st.error(str(exc))
            st.stop()
        except Exception:
            st.error("The question could not be completed. The model or search service may be unavailable. Please try again.")
            st.stop()

    st.divider()
    st.subheader("Answer")
    st.write(state.get("answer", "(no answer produced)"))
    verdict = state.get("generation_verdict")
    if state.get("abstained"):
        st.metric("Evidence", "Not found")
        st.info("No relevant evidence found. Generation and answer-quality grading were skipped.")
    elif verdict:
        first, second, third = st.columns(3)
        first.metric("Sources", str(len(state.get("relevant_chunks", []))))
        second.metric("Source support", "Supported" if verdict["grounded"] else "Needs review")
        third.metric("Question addressed", "Yes" if verdict["addresses_question"] else "Needs review")

    chunks = state.get("relevant_chunks", [])
    if chunks:
        count = len(chunks)
        with st.expander(f"Sources · {count} {'passage' if count == 1 else 'passages'}", expanded=True):
            for index, chunk in enumerate(chunks, 1):
                page = f" — page {chunk['page_number']}" if chunk.get("page_number") else ""
                st.text(f"[{index}] {chunk.get('source_path') or 'Unknown source'}{page} — {chunk.get('section', 'n/a')}")
                st.text(chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else ""))
    with st.expander("Agent trace", expanded=False):
        for line in state.get("trace", []):
            st.text(line)
else:
    st.caption("Answers are limited by the available evidence. Review the sources before relying on a result.")
