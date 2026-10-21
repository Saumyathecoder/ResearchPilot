"""Live single-question policy comparison; web search disabled in both runs."""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from time import perf_counter

import src.llm_client as llm
from src.agent.graph import run_agent
from src.config import CONFIG
from src.retrieval.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path('eval_results') / datetime.now(timezone.utc).strftime('coverage_policy_%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    question = 'In Corrective RAG, what three actions can the system take based on retrieval confidence?'
    report = {'question': question, 'web_enabled': False, 'model': CONFIG.llm_model,
              'top_k': CONFIG.top_k, 'relevance_threshold': CONFIG.relevance_threshold,
              'max_rewrites': CONFIG.max_query_rewrites, 'request_retries': CONFIG.llm_max_retries,
              'limitations': 'One sequential live development question on the existing local sample index. Model verdicts are not independent accuracy labels. Single-run timing is not a benchmark.',
              'runs': {}}
    store = VectorStore()
    report.update(chunk_count=store.count(), embedder=type(store.embedder).__name__)
    original_request = llm._request_once
    original_web, original_coverage = CONFIG.enable_web_fallback, CONFIG.enable_context_coverage
    original_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        CONFIG.enable_web_fallback = False
        for enabled in [False, True]:
            CONFIG.enable_context_coverage = enabled
            calls = [0]
            def counted(*args, **kwargs):
                calls[0] += 1
                return original_request(*args, **kwargs)
            llm._request_once = counted
            started = perf_counter()
            try:
                state = run_agent(question, store=store)
                row = {'status': 'completed', 'requests': calls[0], 'seconds': perf_counter() - started,
                       'rewrite_count': state['rewrite_count'], 'coverage_check_count': state['coverage_check_count'],
                       'verdict': state.get('generation_verdict'), 'answer': state.get('answer'), 'trace': state['trace']}
            except llm.LLMError as exc:
                row = {'status': 'failed', 'error_code': exc.code, 'requests': calls[0]}
            report['runs']['coverage' if enabled else 'ratio_only'] = row
            (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({key: value for key, value in row.items() if key not in {'answer', 'trace', 'verdict'}}), flush=True)
    finally:
        llm._request_once = original_request
        CONFIG.enable_web_fallback, CONFIG.enable_context_coverage = original_web, original_coverage
        logging.disable(original_logging)
    return int(any(row['status'] != 'completed' for row in report['runs'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
