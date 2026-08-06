"""Generates the final answer, grounded in retrieved (and graded-relevant)
context chunks, with inline source attribution."""
from __future__ import annotations

from src.llm_client import complete

_GENERATION_SYSTEM = """You are a technical assistant answering questions \
using ONLY the provided context passages. Rules:

1. Base every claim strictly on the provided context. Do not use outside \
   knowledge, even if you know more about the topic.
2. If the context does not fully answer the question, say so explicitly \
   rather than filling gaps with assumptions.
3. Cite which passage(s) support each part of your answer using the \
   bracketed labels given (e.g. [1], [2]).
4. Be concise and technical. No filler or preamble."""


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        return (
            "I don't have relevant context to answer this question reliably, "
            "and no external source was available either."
        )

    context_block = "\n\n".join(
        f'[{i + 1}] (source: {c.get("source_path", "unknown")}, section: '
        f'{c.get("section", "n/a")})\n{c["text"]}'
        for i, c in enumerate(context_chunks)
    )
    prompt = f"Context passages:\n\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
    return complete(prompt, system=_GENERATION_SYSTEM, temperature=0.0)
