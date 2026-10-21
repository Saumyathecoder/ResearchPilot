# ResearchPilot evaluation — 21 October 2026

## Controlled local retrieval comparison

Both backends used the same 4 public sample documents, 16 chunks, default chunk
settings (800 characters / 120 overlap), exact cosine ranking and k=4. TF-IDF used
the existing fitted SVD backend; MiniLM used the existing sentence-transformer
backend with 384-dimensional normalized vectors. No app index was replaced.

This evaluation made no LLM calls and included no rewriting, grading, generation
or web results. It measures source-document retrieval, not answer accuracy.

| Fixed question set | Backend | Document precision@4 | Document recall@4 | MRR@4 |
|---|---|---:|---:|---:|
| Original: 8 in-scope questions | TF-IDF | 0.46875 | 1.00000 | 1.00000 |
| Original: 8 in-scope questions | MiniLM | 0.56250 | 1.00000 | 1.00000 |
| Paraphrases: 8 in-scope questions | TF-IDF | 0.34375 | 0.87500 | 0.81250 |
| Paraphrases: 8 in-scope questions | MiniLM | 0.40625 | 1.00000 | 0.90625 |

The original set contains one additional out-of-scope query; the paraphrase set
contains two. They are retained in the per-question reports but excluded from
these averages. A forced nearest-neighbor query cannot measure abstention.

### Metric definitions

- **Document precision@4:** proportion of the four returned chunks whose source
  document is in the manually specified expected-document set.
- **Document recall@4:** fraction of expected source documents present among the
  returned chunks, with duplicate chunks from a document counted only once.
- **MRR@4:** mean reciprocal rank of the first returned chunk from an expected
  document; zero if none of the first four chunks comes from an expected document.

Document labels are coarse: a chunk from the expected document may still fail to
answer the question. The scores therefore do not establish passage correctness.

### Improvements and regressions

For “How can a chatbot consult new material without updating its learned
weights?”, TF-IDF missed the expected document in its first four results; MiniLM
ranked it first. This accounts for the paraphrase recall difference.

MiniLM did not improve every query:

- For the fluent-but-wrong response question, precision fell from 0.50 to 0.25,
  and the first expected-document rank worsened from 2 to 4.
- For the meaning-versus-exact-rare-terms question, precision fell from 0.75 to
  0.25, while the first expected-document rank remained 1.

The original questions already had perfect TF-IDF recall and MRR, leaving no room
to improve those metrics. These are small development sets, including manually
authored paraphrases fixed before semantic evaluation. They are not an independent
held-out test or evidence of statistical significance. No tuning was performed
after seeing these results. TF-IDF remains the application's default backend.

### Reproduction and provenance

Run from the repository root using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-semantic.txt
$env:HF_HOME = Join-Path (Get-Location) 'tmp\hf_cache'
.\.venv\Scripts\python.exe -m scripts.benchmark_retrieval --backends tfidf st
.\.venv\Scripts\python.exe -m scripts.benchmark_retrieval --backends tfidf st --dataset data/retrieval_paraphrases.json
```

The first model load downloads public weights. Once cached, setting
`$env:HF_HUB_OFFLINE = '1'` avoids model-hub requests; the paraphrase run used that
setting. Fresh report directories are created and existing outputs are not
overwritten. Raw local reports for the recorded runs are:

- `eval_results/retrieval_original_20261021/results.json`
- `eval_results/retrieval_paraphrases_20261021/results.json`

Model: `sentence-transformers/all-MiniLM-L6-v2`. The downloaded cache's `refs/main`
and snapshot directory resolved to revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The loaded model did not expose its
commit hash through the benchmark's optional model-config probe, so that raw JSON
field is null; this revision was separately verified against the local cache.
Future runs do not pin a revision automatically.

Recorded packages: NumPy 2.4.6, scikit-learn 1.9.1, sentence-transformers 5.7.0,
PyTorch 2.14.1 and transformers 5.20.2. Python: 3.11.9, Windows.

SHA-256 provenance:

| Artifact | SHA-256 |
|---|---|
| Original question dataset | `c3d818d5fdf72335739099cd445dd0d23a04c07bf9a3f3ffc9ce6f133948d7ce` |
| Paraphrase dataset | `ebc74a0fe3211e7bc38639881484d77d5b4bd406726ab8f7d56a8131e55ca191` |
| Serialized chunks, identical in both runs | `297f03580dcdf2ddeff78c1263ef73633794c114dc6953ba6fed99ff24e74c36` |

Single-run timings include import/download/cache effects and are not presented as
a latency benchmark. The model cache, raw evaluation output and private uploads
are ignored by Git. This document is the curated public-data summary.

## Coverage policy: one controlled live comparison

On one Corrective RAG question, both runs retrieved four chunks with one graded
relevant (ratio 0.25 against threshold 0.5). Web search was disabled for both.

| Policy | Provider requests | Query rewrites | Model grading |
|---|---:|---:|---|
| Original relevance-ratio policy | 18 | 2 | Grounded and relevant |
| Evidence-coverage policy | 9 | 0 | Grounded and relevant |

Both runs returned the same answer. The new policy's count includes its additional
coverage check. This is a single observed reduction, not general request savings,
independent answer-quality assessment, or web-search performance. Raw report:
`eval_results/coverage_policy_20261020_v1/results.json`.

The coverage classifier also matched all 6 manually specified synthetic development
examples: three complete-evidence and three partial-evidence cases. That is a
small functional check, not held-out classification accuracy.

## Regression and browser verification

Final publication check: **104 tests passed in 29.38s**;
`pip check` reported no broken requirements. Network/provider failures in the suite
are simulated; the suite does not induce actual outages or prove model accuracy.

Separate live/browser checks verified a PDF-supported MySQL installation answer
with page-1 metadata, unrelated-question abstention with no generation/judge
retries, and coverage-based avoidance of unnecessary recovery. A separate live
synthetic supported-answer/abstention check exercised strict response validation.

The upstream naive-versus-agentic harness was inspected but its initial full live
run did not finish. It mixes raw naive retrieval with filtered/potentially
web-augmented agentic context and uses the inference model as judge. No aggregate
naive-versus-agentic accuracy result is claimed.
