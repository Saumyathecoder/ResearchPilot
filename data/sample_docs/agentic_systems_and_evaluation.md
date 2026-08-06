# Agentic Systems and RAG Evaluation

## What makes a system agentic

A system is considered agentic when a model does not simply produce one
output from one input, but instead makes a sequence of decisions that
determine its own next action: which tool to call, whether to retry, when to
stop, and how to combine intermediate results. An agentic RAG pipeline
therefore differs from a plain RAG pipeline in that the retrieval and
generation steps are wrapped in a control loop with explicit decision
points, rather than being a fixed, single-pass sequence.

## Graph-based orchestration

Representing an agentic pipeline as a directed graph, where each node is a
discrete unit of work (retrieve, grade, rewrite query, generate, check
groundedness) and edges represent conditional transitions, makes the control
flow explicit and testable. Conditional edges can route execution based on
the output of a node, for example routing to a query-rewrite node when a
grading node reports low relevance, and back to retrieval afterward, forming
a bounded retry loop.

## Evaluating RAG systems

Evaluating a RAG system requires separating retrieval quality from
generation quality, because a system can fail at either stage independently.
Retrieval metrics include precision at k (the fraction of retrieved chunks
that are actually relevant) and recall at k (the fraction of all relevant
chunks in the corpus that were retrieved). Generation metrics include
faithfulness (whether the generated answer is entailed by the retrieved
context, typically checked by decomposing the answer into claims and
verifying each claim against the context) and answer relevance (whether the
answer addresses the original question, often estimated by generating
candidate questions from the answer and measuring their similarity to the
original question).

## LLM-as-judge evaluation

When no labeled ground truth is available, a common practical approach is to
use a capable language model as a judge: given the question, the retrieved
context, and the generated answer, the judge model is prompted to output a
structured verdict such as a faithfulness score and a relevance score, along
with a short justification. This approach is not as rigorous as human
annotation but scales far better and correlates reasonably well with human
judgment when the judge prompt is specific and the scoring rubric is
constrained to a small number of discrete categories rather than an open
ended scale.
