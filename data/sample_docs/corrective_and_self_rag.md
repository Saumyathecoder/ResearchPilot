# Corrective RAG and Self-RAG

## Corrective RAG (CRAG)

Corrective RAG introduces a lightweight retrieval evaluator that scores each
retrieved chunk for relevance to the query before generation happens. Based
on the aggregate score, the system takes one of three actions: if confidence
is high, it proceeds to generate using the retrieved chunks as-is; if
confidence is low, it discards the retrieved chunks and falls back to an
external knowledge source such as a web search; if confidence is ambiguous,
it can combine both the retrieved chunks and supplementary web results. CRAG
also applies a decompose-then-recompose step to strip irrelevant sentences
out of chunks that are only partially relevant, rather than treating a chunk
as entirely relevant or entirely irrelevant.

## Self-RAG

Self-RAG trains (or prompts) a model to emit special control decisions
inline with generation: whether retrieval is needed at all for a given
query, whether a retrieved passage is relevant, whether a generated segment
is actually supported by the retrieved passage, and whether the overall
response is useful. This turns retrieval and generation into an interleaved
process with self-critique at each step, rather than a single fixed
retrieve-then-generate pass.

## Why combine both ideas in an agentic system

An agentic RAG system implemented as an explicit graph (rather than a single
prompt) can borrow the retrieval grading idea from CRAG and the generation
self-critique idea from Self-RAG, while keeping each decision as a separate,
inspectable node. This makes the control flow debuggable: a developer can
log exactly which node fired, why a query was rewritten, and why a
generation was rejected and retried, which is difficult to do when all of
this logic is folded into one long prompt.

## Groundedness and answer relevance

Two evaluation questions matter after generation: is the answer grounded
(every claim in the answer is supported by the retrieved context, so the
model did not hallucinate beyond its evidence), and is the answer relevant
(the response actually addresses what the user asked, since a grounded
answer can still fail to address the question if the wrong sub-topic was
retrieved). A robust self-correcting pipeline checks both, independently,
because a system can be grounded-but-irrelevant or relevant-but-ungrounded.
