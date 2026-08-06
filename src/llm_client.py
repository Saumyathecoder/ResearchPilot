"""Thin wrapper around the LLM provider so the rest of the codebase doesn't
repeat model/token/error-handling boilerplate everywhere, and so switching
providers (e.g. paid Anthropic vs free Groq) is a one-line config change
rather than a rewrite.

Set RAG_LLM_PROVIDER=groq (default) or RAG_LLM_PROVIDER=anthropic in .env.
"""
from __future__ import annotations

import json
import logging

from src.config import CONFIG

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazily construct and cache the provider client."""
    global _client
    if _client is not None:
        return _client

    if CONFIG.llm_provider == "groq":
        if not CONFIG.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env, sign up for a free "
                "key at https://console.groq.com/keys, and add it there."
            )
        import groq

        _client = groq.Groq(api_key=CONFIG.groq_api_key)

    elif CONFIG.llm_provider == "anthropic":
        if not CONFIG.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        import anthropic

        _client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)

    else:
        raise ValueError(
            f"Unknown RAG_LLM_PROVIDER: {CONFIG.llm_provider!r}. Use 'groq' or 'anthropic'."
        )

    return _client


def complete(
    prompt: str,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.0,
) -> str:
    """Single-turn completion. Returns plain text. Behavior is identical
    regardless of provider - every caller in this codebase only ever calls
    this function and complete_json(), never a provider SDK directly."""
    client = _get_client()
    max_tokens = max_tokens or CONFIG.llm_max_tokens

    if CONFIG.llm_provider == "groq":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=CONFIG.llm_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    # anthropic
    kwargs = {}
    if system:
        kwargs["system"] = system
    response = client.messages.create(
        model=CONFIG.llm_model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return "".join(block.text for block in response.content if block.type == "text")


def complete_json(
    prompt: str,
    system: str | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Completion that is expected to return strict JSON. Strips code fences
    defensively and raises a clear error if parsing fails, rather than
    silently returning garbage to a caller expecting a dict."""
    json_hint = (
        "\n\nRespond with ONLY the raw JSON object. No markdown code fences, "
        "no preamble, no explanation."
    )
    raw = complete(
        prompt, system=(system or "") + json_hint, max_tokens=max_tokens, temperature=0.0
    )
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1) if cleaned.startswith("json\n") else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse JSON from LLM output: %s", raw)
        raise ValueError(f"LLM did not return valid JSON: {exc}\nRaw output: {raw}") from exc
