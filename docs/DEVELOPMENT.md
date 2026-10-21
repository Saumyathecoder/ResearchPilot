# Development guide

Use Python 3.11 and the project virtual environment. Install core dependencies
from `requirements.txt`; semantic embeddings additionally need
`requirements-semantic.txt` and a first-run model download.

## Code layout

| Path | Responsibility |
|---|---|
| `app.py`, `assets/app.css`, `.streamlit/config.toml` | Streamlit workspace and dark theme |
| `src/ingestion` | Text/PDF loading, chunking and session-owned upload indexes |
| `src/retrieval` | TF-IDF / sentence-transformer embedders and persistent Chroma |
| `src/graders` | Routing, relevance, coverage, rewriting and generation checks |
| `src/agent` | State, nodes and bounded graph transitions |
| `src/llm_client.py` | Provider requests, retry budget and strict response validation |
| `src/evaluation`, `scripts` | Retrieval and live coverage evaluations |
| `tests` | Offline regression suite, including Streamlit wiring |

## Working with indexes

The sample index and fitted embedder live under `chroma_db/`. Rebuild deliberately
when changing the corpus or embedding backend. Refitting into a populated collection
without rebuilding is rejected because the old vectors would use a different space.

Uploads use an in-memory index owned by one Streamlit session. Build a replacement
completely before assigning it to session state. A failed replacement leaves the
previous collection active. Never put uploaded text in a shared Streamlit cache.

PDF pages remain separate through chunking. Chunk metadata includes filename,
document ID, section and optional 1-based page number. Passage IDs include the page
number so multiple pages cannot overwrite each other.

## Agent decisions

Sample questions are routed before retrieval. Uploaded questions retrieve directly
against the session corpus. Retrieval graders always receive the original question,
even after rewriting.

With a low relevance ratio but nonempty evidence, coverage checks whether every
requested part is supported. Only a validated positive result bypasses recovery.
Failed checks keep the original ratio policy. High-ratio batches and empty evidence
do not incur a coverage call. Set `RAG_ENABLE_CONTEXT_COVERAGE=false` for an ablation.

If all permitted recovery finishes without relevant evidence, the graph abstains
without generation or judge calls. Generated answers still receive both grounding
and answer-relevance checks, with bounded retries and a caveat on exhaustion.

## Model requests

The adapter owns one retry budget; SDK retries are disabled. Default request timeout
is 30 seconds and retry count is one. Transport, empty-output, JSON and schema
recovery share that budget. Long rate-limit waits surface an error rather than
retrying immediately. Token-limit finishes are rejected rather than presented as
complete answers.

JSON responses must be objects with correctly typed fields. Duplicate keys,
non-finite values, string booleans, unknown routes and contradictory verdicts are
rejected. Do not add default passing judgments. Logs and public errors must not
contain raw model output, provider bodies, credentials or private document text.

The Groq GPT-OSS path preserves a tested 4096-token minimum and low reasoning
setting. This is a compatibility choice, not a confirmed diagnosis of an earlier
empty response. References: [Groq SDK](https://github.com/groq/groq-python#retries),
[reasoning parameters](https://console.groq.com/docs/reasoning).

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Provider failures are simulated in tests. Browser checks should cover a supported
PDF answer with page metadata, an unrelated question that abstains, failed upload
replacement, and a new session with no access to an existing session's uploads.
Keep the sample demo usable after upload changes.

Evaluation commands and measured results are in [EVALUATION.md](EVALUATION.md).
Run output directories are created fresh; existing reports are never overwritten.

## Files that stay local

Keep `.env`, local secret variants, `.streamlit/secrets.toml`, `.venv/`, `tmp/`,
`chroma_db/`, `uploads/` and raw `eval_results/` out of version control. The model
cache can be placed under ignored `tmp/hf_cache` with `HF_HOME`.

Publish only public sample data, synthetic test fixtures and curated evaluation
summaries. Check screenshot contents before adding them. Component attribution is
listed in [CONTRIBUTIONS.md](CONTRIBUTIONS.md); preserve the MIT notice in `LICENSE`.
