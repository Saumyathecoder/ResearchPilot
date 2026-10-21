"""Small live development-set check for context coverage, not answer accuracy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from src.config import CONFIG
from src.graders.coverage_grader import check_context_coverage
from src.llm_client import LLMError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path('eval_results') / datetime.now(timezone.utc).strftime('coverage_%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    data = Path('data/coverage_eval_dataset.json').read_bytes()
    rows = []
    report = {'dataset_sha256': hashlib.sha256(data).hexdigest(),
              'provider': CONFIG.llm_provider, 'model': CONFIG.llm_model,
              'dataset_kind': 'six synthetic development examples with manually specified expected coverage',
              'limitations': 'No held-out accuracy, latency, call-reduction or end-to-end quality claim. No web results.',
              'per_case': rows}
    for case in json.loads(data):
        row = {'id': case['id'], 'expected_sufficient': case['expected_sufficient']}
        try:
            verdict = check_context_coverage(case['question'], [{'text': text} for text in case['passages']])
            row.update(status='completed', verdict=verdict, correct=verdict['sufficient'] == case['expected_sufficient'])
        except LLMError as exc:
            row.update(status='failed', error_code=exc.code)
        rows.append(row)
        report['completed'] = sum(r['status'] == 'completed' for r in rows)
        report['correct'] = sum(r.get('correct', False) for r in rows)
        report['false_sufficient'] = sum(not r['expected_sufficient'] and r.get('verdict', {}).get('sufficient', False) for r in rows)
        (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f"{case['id']}: {row.get('correct', row['status'])}", flush=True)
    print(f'Report saved to {output}', flush=True)
    return int(report['completed'] != len(rows) or report['correct'] != len(rows))


if __name__ == '__main__':
    raise SystemExit(main())
