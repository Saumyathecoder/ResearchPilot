# Embeddings and Vector Search

## What an embedding is

An embedding is a fixed-length numeric vector that represents a piece of
text such that semantically similar text maps to nearby points in the
vector space. Embeddings can be produced by dense neural encoders (such as
sentence-transformer models) or by sparse statistical methods (such as
TF-IDF), and different embedding methods trade off quality, latency, and
infrastructure cost differently.

## Dense vs sparse retrieval

Dense retrieval uses neural embeddings and cosine or dot-product similarity,
and tends to capture semantic similarity even when exact words differ.
Sparse retrieval methods such as BM25 or TF-IDF match on lexical overlap and
term statistics, and tend to do well on queries with rare or specific terms
such as product codes, names, or exact phrases that a dense encoder may
smooth over. Hybrid retrieval systems combine both signals, typically by
computing a weighted sum or a reciprocal rank fusion of the two rankings, to
get the benefits of each.

## Vector databases

A vector database indexes embeddings so that nearest-neighbor search can run
in sublinear time on large collections, using approximate nearest neighbor
algorithms such as HNSW (hierarchical navigable small world graphs) or IVF
(inverted file index). For small to medium corpora (up to roughly a few
hundred thousand chunks), exact search is often fast enough that the choice
of index matters less than the choice of embedding model and chunking
strategy.

## Chunking strategy

How a document is split into chunks materially affects retrieval quality.
Chunks that are too large dilute the embedding with multiple topics, making
similarity search less precise. Chunks that are too small lose surrounding
context needed to make sense of a passage. Overlapping chunks (where
consecutive chunks share a sliding window of text) help preserve context
across chunk boundaries at the cost of redundant storage. Structure-aware
chunking, which splits along natural document boundaries such as headings,
paragraphs, or function definitions in code, generally outperforms naive
fixed-length splitting.
