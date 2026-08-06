import textwrap

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import RawDocument, load_documents


def test_load_documents_reads_sample_corpus():
    docs = load_documents("data/sample_docs")
    assert len(docs) >= 4
    assert all(doc.text for doc in docs)
    assert all(doc.doc_id for doc in docs)


def test_load_documents_missing_dir_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does_not_exist")


def test_chunker_respects_headings_and_produces_nonempty_chunks():
    text = textwrap.dedent(
        """
        # Heading One
        Paragraph one under heading one.

        Paragraph two under heading one.

        # Heading Two
        Paragraph one under heading two.
        """
    ).strip()
    doc = RawDocument(doc_id="test_doc", source_path="test.md", text=text)
    chunks = chunk_documents([doc], chunk_size=1000, overlap=0)

    assert len(chunks) == 2
    sections = {c.section for c in chunks}
    assert sections == {"Heading One", "Heading Two"}
    assert all(c.doc_id == "test_doc" for c in chunks)
    # chunk ids must be unique
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunker_splits_long_sections_into_multiple_chunks():
    long_paragraph = "Sentence about RAG systems and retrieval. " * 40
    text = f"# Long Section\n\n{long_paragraph}\n\n{long_paragraph}"
    doc = RawDocument(doc_id="long_doc", source_path="long.md", text=text)
    chunks = chunk_documents([doc], chunk_size=300, overlap=50)

    assert len(chunks) > 1
    assert all(len(c.text) <= 300 + 50 + 5 for c in chunks)  # allow small slack for overlap join
