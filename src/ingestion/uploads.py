"""Session-owned upload corpus. No uploaded content or index is written to disk."""
from dataclasses import asdict

import numpy as np

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import DocumentError, load_document_bytes
from src.retrieval.embeddings import get_embedder

MAX_FILES = 10
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 30 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000


class UploadCorpus:
    """Build completely before replacing a session's previous corpus."""

    def __init__(self, files: list[tuple[str, bytes]]):
        if not files or len(files) > MAX_FILES:
            raise DocumentError(f"Select between 1 and {MAX_FILES} files.")
        if sum(len(data) for _, data in files) > MAX_TOTAL_BYTES:
            raise DocumentError("Uploads exceed the 30 MB total limit.")
        documents = []
        self.warnings = []
        names = set()
        filenames = []
        for filename, data in files:
            name = filename.replace("\\", "/").rsplit("/", 1)[-1]
            if name.casefold() in names:
                raise DocumentError("Duplicate filenames are ambiguous. Rename files before uploading.")
            names.add(name.casefold())
            filenames.append(name)
            if len(data) > MAX_FILE_BYTES:
                raise DocumentError(f"{name}: exceeds the 10 MB per-file limit.")
            documents.extend(load_document_bytes(name, data, warnings=self.warnings))
        if sum(len(doc.text) for doc in documents) > MAX_TEXT_CHARS:
            raise DocumentError("Extracted text exceeds the 2 million character limit.")
        self.chunks = [asdict(chunk) for chunk in chunk_documents(documents)]
        if not self.chunks:
            raise DocumentError("No indexable text was found in these documents.")
        self.embedder = get_embedder()
        try:
            self.embedder.fit([chunk['text'] for chunk in self.chunks])
            self.vectors = self.embedder.embed([chunk['text'] for chunk in self.chunks])
        except ValueError:
            raise DocumentError("Text could not be indexed. Include meaningful words rather than only punctuation or stop words.") from None
        self.filenames = sorted(filenames)
        self.file_count = len(files)

    def count(self) -> int:
        return len(self.chunks)

    def similarity_search(self, query: str, k: int = 4) -> list[dict]:
        vector = self.embedder.embed([query])[0]
        scores = self.vectors @ vector
        indices = np.argsort(-scores, kind="stable")[:k]
        return [{**self.chunks[i], 'similarity': max(0.0, float(scores[i]))} for i in indices]
