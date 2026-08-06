"""Structure-aware chunking for markdown/text documents.

Naive fixed-length splitting cuts through sentences and headings. This
chunker first splits along markdown headings and paragraph boundaries, then
packs paragraphs into chunks up to `chunk_size` characters with a sliding
overlap, so chunk boundaries respect document structure whenever possible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import CONFIG
from src.ingestion.loader import RawDocument

_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    section: str  # nearest preceding heading, for traceability/citations


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (heading, body) sections using markdown headings."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("(no heading)", text)]

    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        heading = match.group().lstrip("#").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _hard_split(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a single paragraph that alone exceeds chunk_size, on word
    boundaries, with a sliding character overlap between pieces."""
    words = paragraph.split(" ")
    pieces: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + len(word) + 1 <= chunk_size:
            current += " " + word
        else:
            pieces.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + " " + word).strip()
    if current:
        pieces.append(current)
    return pieces


def _pack_paragraphs(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into ~chunk_size chunks with character
    overlap. Paragraphs that alone exceed chunk_size are hard-split on word
    boundaries first, so no returned chunk is ever wildly over budget."""
    normalized: list[str] = []
    for para in paragraphs:
        if len(para) > chunk_size:
            normalized.extend(_hard_split(para, chunk_size, overlap))
        else:
            normalized.append(para)

    chunks: list[str] = []
    current = ""
    for para in normalized:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= chunk_size:
            current += "\n\n" + para
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + "\n\n" + para).strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(
    documents: list[RawDocument],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Chunk a list of RawDocuments into overlapping, section-aware Chunks."""
    chunk_size = chunk_size or CONFIG.chunk_size
    overlap = overlap or CONFIG.chunk_overlap

    all_chunks: list[Chunk] = []
    for doc in documents:
        sections = _split_into_sections(doc.text)
        chunk_index = 0
        for heading, body in sections:
            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            packed = _pack_paragraphs(paragraphs, chunk_size, overlap)
            for piece in packed:
                all_chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::{chunk_index}",
                        doc_id=doc.doc_id,
                        source_path=doc.source_path,
                        text=piece,
                        section=heading,
                    )
                )
                chunk_index += 1
    return all_chunks
