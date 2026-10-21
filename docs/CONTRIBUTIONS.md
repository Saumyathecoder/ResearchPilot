# Component provenance

ResearchPilot uses the MIT-licensed [hejerr/agentic-rag](https://github.com/hejerr/agentic-rag)
codebase as its starting point. The retained baseline is commit
`a8f69312e69c2d978b0cdd0edaf1902e1982920b`.

| Component | Provenance / changes |
|---|---|
| LangGraph pipeline, initial routing and graders, query rewriting, web fallback | Retained core; added faithful rewrites, coverage-aware recovery, strict validation and explicit abstention |
| TF-IDF and sentence-transformer backends, Chroma persistence | Retained backends; added tiny-corpus handling and safer indexing guards |
| Markdown/text ingestion and numbered passage citations | Retained core; added PDF loading, page metadata, file validation and session-isolated multi-file uploads |
| Provider adapter | Added bounded requests/retries, typed safe errors, empty/truncated-output handling and strict schemas; retained Groq/Anthropic support |
| Initial evaluation harness and sample corpus | Retained; added controlled retrieval comparison, paraphrase probes, coverage fixtures and measured policy comparison |
| Streamlit interface | Custom ResearchPilot dark workspace, document controls, source display, plain-language statuses and optional trace |
| Tests and documentation | Expanded regressions, reproducible evaluation reports, screenshot and project documentation |

Project maintenance and ResearchPilot changes: [Saumya Joshi](https://github.com/Saumyathecoder).
The original MIT copyright and permission notice remain in [LICENSE](../LICENSE).
