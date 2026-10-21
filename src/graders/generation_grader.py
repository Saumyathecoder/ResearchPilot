"""Post-generation self-critique: checks groundedness (is every claim
supported by the retrieved context?) and answer relevance (does the answer
actually address the question?) independently, since a system can fail
either check without failing the other.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.llm_client import LLMError, complete_json, validate_fields

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


def _validate_groundedness(result: dict) -> None:
    validate_fields(result, {'grounded': bool, 'unsupported_claims': list, 'reason': str})
    if result['grounded'] and result['unsupported_claims']:
        raise LLMError('invalid_schema', 'The model returned contradictory grounding checks. Please try again.', retryable=True)


def grade_generation(question: str, context_chunks: list[dict], answer: str) -> GenerationVerdict:
    context_block = "\n\n".join(c["text"] for c in context_chunks) or "(no context provided)"

    grounded_result = complete_json(
        f"Context passages:\n\n{context_block}\n\nGenerated answer:\n{answer}",
        system=_GROUNDEDNESS_SYSTEM,
        max_tokens=300,
        validator=_validate_groundedness,
    )
    relevance_result = complete_json(
        f'Original question: "{question}"\n\nGenerated answer:\n{answer}',
        system=_RELEVANCE_SYSTEM,
        max_tokens=150,
        validator=lambda r: validate_fields(r, {'addresses_question': bool, 'reason': str}),
    )

    reasons = [grounded_result['reason'], relevance_result['reason']]
    return GenerationVerdict(
        grounded=grounded_result['grounded'],
        addresses_question=relevance_result['addresses_question'],
        unsupported_claims=grounded_result['unsupported_claims'],
        reason=" | ".join(r for r in reasons if r),
    )
