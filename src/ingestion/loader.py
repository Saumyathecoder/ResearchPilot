"""Load raw documents from disk into a normalized in-memory representation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass
class RawDocument:
    """A single loaded document before chunking."""

    doc_id: str
    source_path: str
    text: str


def load_documents(directory: str | Path) -> list[RawDocument]:
    """Load every supported file in `directory` into RawDocument objects.

    Parameters
    ----------
    directory:
        Path to a folder containing .md / .txt files.

    Returns
    -------
    List of RawDocument, one per file, sorted by filename for determinism.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    paths = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        logger.warning("No supported documents (%s) found in %s", SUPPORTED_EXTENSIONS, directory)

    documents: list[RawDocument] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            logger.warning("Skipping empty document: %s", path)
            continue
        documents.append(RawDocument(doc_id=path.stem, source_path=str(path), text=text))

    logger.info("Loaded %d documents from %s", len(documents), directory)
    return documents
