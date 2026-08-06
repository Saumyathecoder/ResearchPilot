"""Post-generation self-critique: checks groundedness (is every claim
supported by the retrieved context?) and answer relevance (does the answer
actually address the question?) independently, since a system can fail
either check without failing the other.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.llm_client import complete_json

_GROUNDEDNESS_SYSTEM = """You are a strict fact-checker. Given a set of \
context passages and a generated answer, determine whether the answer's \
claims are all supported by the passages (grounded), or whether it \
contains claims not backed by the passages (hallucinated).

Respond with ONLY a JSON object: \
{"grounded": true|false, "unsupported_claims": ["..."], "reason": "<one short sentence>"}"""

_RELEVANCE_SYSTEM = """You judge whether a generated answer actually \
addresses the user's original question, regardless of whether the answer \
is factually correct. An answer can be well-grounded in its sources but \
still fail to address what was actually asked.

Respond with ONLY a JSON object: {"addresses_question": true|false, "reason": "<one short sentence>"}"""


@dataclass
class GenerationVerdict:
    grounded: bool
    addresses_question: bool
    unsupported_claims: list[str]
    reason: str

    @property
    def passed(self) -> bool:
        return self.grounded and self.addresses_question


def grade_generation(question: str, context_chunks: list[dict], answer: str) -> GenerationVerdict:
    context_block = "\n\n".join(c["text"] for c in context_chunks) or "(no context provided)"

    grounded_result = complete_json(
        f"Context passages:\n\n{context_block}\n\nGenerated answer:\n{answer}",
        system=_GROUNDEDNESS_SYSTEM,
        max_tokens=300,
    )
    relevance_result = complete_json(
        f'Original question: "{question}"\n\nGenerated answer:\n{answer}',
        system=_RELEVANCE_SYSTEM,
        max_tokens=150,
    )

    reasons = [grounded_result.get("reason", ""), relevance_result.get("reason", "")]
    return GenerationVerdict(
        grounded=bool(grounded_result.get("grounded", False)),
        addresses_question=bool(relevance_result.get("addresses_question", False)),
        unsupported_claims=grounded_result.get("unsupported_claims", []) or [],
        reason=" | ".join(r for r in reasons if r),
    )
