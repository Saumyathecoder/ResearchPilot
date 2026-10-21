"""Run: python -m scripts.benchmark_retrieval --backends tfidf st

Reports raw local retrieval only, independent of the upstream live evaluation.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import re
from pathlib import Path
from time import perf_counter

from src.config import CONFIG
from src.evaluation.retrieval_benchmark import evaluate_retriever
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents
from src.retrieval.embeddings import get_embedder


def model_revision(embedder):
    """Best-effort provenance from the loaded model; never fetch extra files."""
    model = getattr(embedder, 'model', None)
    if model is not None:
        for module in model.modules():
            revision = getattr(getattr(module, 'config', None), '_commit_hash', None)
            if isinstance(revision, str) and re.fullmatch(r'[0-9a-f]{40,64}', revision):
                return revision
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backends', nargs='+', choices=['tfidf', 'st'], default=['tfidf'])
    parser.add_argument('--corpus', type=Path, default=Path('data/sample_docs'))
    parser.add_argument('--dataset', type=Path, default=Path('data/eval_dataset.json'))
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.k < 1:
        parser.error('--k must be positive')
    output = args.output or Path('eval_results') / datetime.now(timezone.utc).strftime('retrieval_%Y%m%dT%H%M%S%fZ')
    # Refuse overwrite, including earlier baseline results.
    output.mkdir(parents=True, exist_ok=False)
    dataset_bytes = args.dataset.read_bytes()
    questions = json.loads(dataset_bytes)
    chunks = chunk_documents(load_documents(args.corpus))
    snapshot = json.dumps([asdict(chunk) for chunk in chunks], sort_keys=True).encode()
    packages = {}
    for name in ['numpy', 'scikit-learn', 'sentence-transformers', 'torch', 'transformers']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {'protocol': 'local-retrieval-v1', 'k': args.k,
              'chunk_count': len(chunks), 'document_count': len({c.doc_id for c in chunks}),
              'chunk_size': CONFIG.chunk_size, 'chunk_overlap': CONFIG.chunk_overlap,
              'dataset_sha256': hashlib.sha256(dataset_bytes).hexdigest(),
              'chunks_sha256': hashlib.sha256(snapshot).hexdigest(),
              'semantic_model': CONFIG.st_model_name, 'packages': packages,
              'limitations': ['Document-level labels do not establish passage-level relevance.',
                              'Out-of-scope queries excluded from precision, recall and MRR averages.',
                              'No web results, LLM grading, rewriting, generation or answer-accuracy metrics.',
                              'Small upstream dataset; no statistical significance or generalization claim.',
                              'Timing is single-run and includes warm-cache effects.'],
              'backends': {}}
    for backend in dict.fromkeys(args.backends):
        print(f'Evaluating {backend}...', flush=True)
        started = perf_counter()
        try:
            embedder = get_embedder(backend)
            load_seconds = perf_counter() - started
            result = evaluate_retriever(chunks, questions, embedder, k=args.k)
            result.update(status='completed', model_load_seconds=load_seconds,
                          model_revision=model_revision(embedder))
        except Exception as exc:
            # Never copy provider/config values or arbitrary exception messages.
            result = {'status': 'failed', 'error_type': type(exc).__name__}
        report['backends'][backend] = result
        (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(backend, result.get('summary', result['status']), flush=True)
    lines = ['# Local retrieval comparison', '',
             f"k={args.k}; {report['document_count']} documents; {len(chunks)} chunks.", '',
             '| Backend | Precision | Recall | MRR |', '|---|---:|---:|---:|']
    for backend, result in report['backends'].items():
        if result['status'] == 'completed':
            s = result['summary']
            values = ['n/a' if s[key] is None else f'{s[key]:.4f}' for key in
                      ['mean_precision', 'mean_recall', 'mean_reciprocal_rank']]
            lines.append(f"| {backend} | {' | '.join(values)} |")
        else:
            lines.append(f"| {backend} | failed ({result['error_type']}) | — | — |")
    lines.extend(['', *report['limitations']])
    (output / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Results saved to {output}')
    return int(any(result['status'] != 'completed' for result in report['backends'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
