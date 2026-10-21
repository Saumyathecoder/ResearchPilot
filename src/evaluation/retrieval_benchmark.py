"""Controlled local retrieval comparison. No LLM judge, router, or web search."""
from __future__ import annotations

from time import perf_counter

import numpy as np

from src.evaluation.metrics import retrieval_precision_at_k, retrieval_recall_at_k


def evaluate_retriever(chunks, questions, embedder, *, k=4):
    """Evaluate identical chunks and queries with document-level relevance labels.

    Out-of-scope questions are reported separately: forced nearest-neighbor
    retrieval is not an abstention system. MRR uses the first expected document.
    """
    if not chunks or k < 1:
        raise ValueError('Provide nonempty chunks and a positive k.')
    doc_ids = {chunk.doc_id for chunk in chunks}
    for item in questions:
        if not isinstance(item.get('question'), str) or not item['question'].strip():
            raise ValueError('Every question must be nonempty text.')
        expected = item.get('expected_doc_ids')
        if not isinstance(expected, list) or not all(isinstance(d, str) for d in expected):
            raise ValueError('expected_doc_ids must be a list of document IDs.')
        if len(expected) != len(set(expected)) or set(expected) - doc_ids:
            raise ValueError('Expected document IDs must be unique and present in the corpus.')

    started = perf_counter()
    embedder.fit([chunk.text for chunk in chunks])
    vectors = np.asarray(embedder.embed([chunk.text for chunk in chunks]))
    fit_seconds = perf_counter() - started
    if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
        raise ValueError('Invalid corpus embeddings.')
    # Explicit cosine normalization supports either existing backend.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    rows = []
    for item in questions:
        started = perf_counter()
        query = np.asarray(embedder.embed([item['question']]))[0]
        if query.shape != (vectors.shape[1],) or not np.isfinite(query).all():
            raise ValueError('Query and corpus embedding dimensions must match.')
        norm = np.linalg.norm(query)
        scores = vectors @ (query / norm if norm else query)
        selected = np.argsort(-scores, kind='stable')[:k]
        elapsed = perf_counter() - started
        hits = [{'doc_id': chunks[i].doc_id, 'chunk_id': chunks[i].chunk_id,
                 'score': float(scores[i])} for i in selected]
        expected = item['expected_doc_ids']
        row = {'question': item['question'], 'expected_doc_ids': expected,
               'hits': hits, 'query_seconds': elapsed, 'in_scope': bool(expected)}
        if expected:
            row.update(precision=retrieval_precision_at_k(hits, expected),
                       recall=retrieval_recall_at_k(hits, expected),
                       reciprocal_rank=next((1 / rank for rank, hit in enumerate(hits, 1)
                                             if hit['doc_id'] in expected), 0.0))
        rows.append(row)
    included = [row for row in rows if row['in_scope']]
    summary = {f'mean_{key}': float(np.mean([row[key] for row in included])) if included else None
               for key in ['precision', 'recall', 'reciprocal_rank']}
    return {'summary': summary, 'embedding_dimension': vectors.shape[1], 'in_scope_questions': len(included),
            'out_of_scope_questions': len(rows) - len(included),
            'fit_and_corpus_embedding_seconds': fit_seconds, 'per_question': rows}
