# Retrieval-Augmented Generation: Fundamentals

Retrieval-Augmented Generation (RAG) is an architecture that combines a
parametric language model with a non-parametric retrieval component. Instead
of relying only on knowledge memorized during pretraining, a RAG system
fetches relevant passages from an external corpus at inference time and
conditions the generation step on that retrieved evidence.

## Why RAG exists

Large language models have three structural weaknesses that RAG addresses:

1. **Knowledge cutoff** - a model cannot know about documents or events that
   did not exist during training.
2. **Hallucination** - when a model is uncertain, it tends to produce fluent
   but unsupported text rather than admitting it does not know.
3. **Auditability** - a pure parametric answer cannot be traced back to a
   source, which is a problem in regulated or high-stakes domains.

By grounding generation in retrieved text, a RAG pipeline can cite sources,
stay current without retraining, and reduce (but not eliminate) hallucination.

## The naive RAG pipeline

A minimal RAG system has four stages: chunking, embedding, retrieval, and
generation. Documents are split into chunks, each chunk is embedded into a
vector space, a user query is embedded into the same space, the top-k most
similar chunks are retrieved by a distance metric such as cosine similarity,
and those chunks are inserted into the prompt for the generator.

## Known failure modes of naive RAG

Naive RAG assumes retrieval always returns useful context. In practice,
retrieval can fail silently: the top-k chunks may be topically related but
not actually answer the question, the corpus may simply not contain the
answer, or the query may be phrased in a way that does not match how the
answer is phrased in the corpus (vocabulary mismatch). When this happens, a
naive pipeline still generates an answer, often a confident but incorrect
one, because the generation step has no mechanism to evaluate the quality of
what it was given.

This is the core motivation for corrective and agentic RAG architectures,
which add explicit evaluation and control-flow steps around retrieval.
