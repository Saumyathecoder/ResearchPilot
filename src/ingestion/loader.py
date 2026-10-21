"""Load raw documents from disk into a normalized in-memory representation."""
from __future__ import annotations

import logging
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


@dataclass
class RawDocument:
    """A single loaded document before chunking."""

    doc_id: str
    source_path: str
    text: str
    page_number: int | None = None


class DocumentError(ValueError):
    """A document cannot be safely read as supported text."""


def load_document_bytes(filename: str, content: bytes, *, doc_id: str | None = None,
                        warnings: list[str] | None = None) -> list[RawDocument]:
    """Read UTF-8 text or PDF pages without persisting uploaded bytes. No OCR."""
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentError("Unsupported file type. Use PDF, Markdown, or plain text.")
    if not content:
        raise DocumentError("The file is empty.")
    doc_id = doc_id or hashlib.sha256(filename.encode() + b"\0" + content).hexdigest()
    if suffix != ".pdf":
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError:
            raise DocumentError("Text files must use UTF-8 encoding.") from None
        if not text or "\0" in text:
            raise DocumentError("The file contains no usable text.")
        return [RawDocument(doc_id, filename, text)]

    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise DocumentError("Encrypted PDFs are not supported. Upload an unencrypted copy.")
        if len(reader.pages) > 200:
            raise DocumentError("PDFs are limited to 200 pages per file.")
        documents = []
        skipped = []
        for number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                documents.append(RawDocument(doc_id, filename, text, number))
            else:
                skipped.append(number)
        if not documents:
            raise DocumentError("PDF has no extractable text. Scanned/image-only PDFs need OCR, which is not supported.")
        if skipped and warnings is not None:
            warnings.append(f"{filename}: skipped pages without extractable text: {', '.join(map(str, skipped))}. No OCR was performed.")
        return documents
    except DocumentError:
        raise
    except Exception:
        raise DocumentError("PDF could not be read. It may be corrupt or unsupported.") from None


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
        if path.stat().st_size == 0:
            logger.warning("Skipping empty document: %s", path)
            continue
        # Preserve IDs used by the upstream evaluation for top-level sample files;
        # relative names distinguish same-stem files in nested directories.
        identity = path.relative_to(directory).with_suffix("").as_posix()
        if sum(p.relative_to(directory).with_suffix("").as_posix() == identity for p in paths) > 1:
            identity = path.relative_to(directory).as_posix()
        loaded = load_document_bytes(path.name, path.read_bytes(), doc_id=identity)
        for doc in loaded:
            doc.source_path = str(path)
        documents.extend(loaded)

    logger.info("Loaded %d documents from %s", len(documents), directory)
    return documents
