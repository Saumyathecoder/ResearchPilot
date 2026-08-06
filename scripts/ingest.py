"""Build (or rebuild) the vector store from data/sample_docs.

Usage:
    python -m scripts.ingest
    python -m scripts.ingest --docs-dir path/to/other/docs --rebuild
"""
from __future__ import annotations

import argparse
import logging

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents
from src.retrieval.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", default="data/sample_docs")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and rebuild the collection")
    args = parser.parse_args()

    documents = load_documents(args.docs_dir)
    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents)} documents -> {len(chunks)} chunks")

    store = VectorStore()
    store.index_chunks(chunks, rebuild=args.rebuild)
    print(f"Indexed. Collection now has {store.count()} chunks.")


if __name__ == "__main__":
    main()
