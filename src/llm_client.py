"""Provider adapter with a single retry budget and secret-safe failures."""
from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Callable

from src.config import CONFIG

logger = logging.getLogger(__name__)
_client = None


class LLMError(RuntimeError):
    """Safe, actionable errors; never include provider bodies or model output."""

    def __init__(self, code: str, message: str, *, retryable: bool = False, delay: float = .5):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.delay = delay


def _settings() -> tuple[float, int]:
    timeout, retries = CONFIG.llm_timeout_seconds, CONFIG.llm_max_retries
    if not math.isfinite(timeout) or not 1 <= timeout <= 120 or not 0 <= retries <= 2:
        raise LLMError('configuration', 'Model timeout must be 1–120 seconds and retries 0–2. Check configuration.')
    return timeout, retries


def _get_client():
    global _client
    if _client is not None:
        return _client
    timeout, _ = _settings()
    # Disable SDK retries: the shared operation loop owns the full retry budget.
    if CONFIG.llm_provider == 'groq':
        if not CONFIG.groq_api_key:
            raise LLMError('configuration', 'No Groq API key configured. Add it to your local environment or app secrets.')
        import groq
        _client = groq.Groq(api_key=CONFIG.groq_api_key, timeout=timeout, max_retries=0)
    elif CONFIG.llm_provider == 'anthropic':
        if not CONFIG.anthropic_api_key:
            raise LLMError('configuration', 'No Anthropic API key configured. Add it to your local environment or app secrets.')
        import anthropic
        _client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key, timeout=timeout, max_retries=0)
    else:
        raise LLMError('configuration', 'Unknown model provider. Choose groq or anthropic in configuration.')
    return _client


def _api_error(exc: Exception) -> LLMError:
    import groq
    import anthropic
    if isinstance(exc, (groq.APITimeoutError, anthropic.APITimeoutError)):
        return LLMError('timeout', 'The model request timed out. Please try again.', retryable=True)
    if isinstance(exc, (groq.APIConnectionError, anthropic.APIConnectionError)):
        return LLMError('connection', 'Could not connect to the model service. Check your connection and try again.', retryable=True)
    status = getattr(exc, 'status_code', None)
    if status in {401, 403}:
        return LLMError('authentication', 'The model service rejected access. Check your API key and model permissions.')
    if status == 429:
        headers = getattr(getattr(exc, 'response', None), 'headers', {})
        after = headers.get('retry-after')
        try:
            delay = float(after) if after is not None else 1.0
        except (ValueError, TypeError):
            delay = math.inf
        short_wait = math.isfinite(delay) and 0 <= delay <= 5
        return LLMError('rate_limit', 'The model service rate limit was reached. Please wait before trying again.',
                        retryable=short_wait, delay=delay if short_wait else 0)
    if status in {408, 409} or isinstance(status, int) and status >= 500:
        return LLMError('unavailable', 'The model service is temporarily unavailable. Please try again.', retryable=True)
    return LLMError('request_failed', 'The model request failed. Check the configured model and provider settings.')


def _request_once(prompt, system, max_tokens, temperature) -> str:
    client = _get_client()
    max_tokens = max_tokens or CONFIG.llm_max_tokens
    try:
        if CONFIG.llm_provider == 'groq':
            messages = ([{'role': 'system', 'content': system}] if system else [])
            messages.append({'role': 'user', 'content': prompt})
            groq_options = {}
            # Preserve the proven local GPT-OSS compatibility adjustment.
            if CONFIG.llm_model.startswith('openai/gpt-oss'):
                max_tokens = max(max_tokens, 4096)
                groq_options['reasoning_effort'] = 'low'
            response = client.chat.completions.create(
                model=CONFIG.llm_model, messages=messages,
                max_completion_tokens=max_tokens, temperature=temperature, **groq_options)
            choice = response.choices[0] if response.choices else None
            reason = getattr(choice, 'finish_reason', None)
            text = choice.message.content if choice else None
            truncated = reason == 'length'
        else:
            kwargs = {'system': system} if system else {}
            response = client.messages.create(
                model=CONFIG.llm_model, max_tokens=max_tokens, temperature=temperature,
                messages=[{'role': 'user', 'content': prompt}], **kwargs)
            reason = response.stop_reason
            text = ''.join(block.text for block in response.content if block.type == 'text')
            truncated = reason == 'max_tokens'
    except LLMError:
        raise
    except Exception as exc:
        raise _api_error(exc) from None

    # Only known enum values are logged, never provider text or reasoning.
    safe_reason = reason if reason in {'stop', 'length', 'content_filter', 'end_turn', 'max_tokens', 'tool_calls', 'tool_use'} else 'other'
    logger.debug('Model completion finish=%s', safe_reason)
    if truncated:
        raise LLMError('truncated', 'The model reached its output-token limit. Try a narrower question or increase the configured token limit.')
    if reason not in {'stop', 'end_turn'}:
        raise LLMError('incomplete', 'The model did not return a completed text response. Please try again.', retryable=reason is None)
    if not isinstance(text, str) or not text.strip():
        raise LLMError('empty_output', 'The model returned an empty response. Please try again.', retryable=True)
    return text.strip()


def _run(prompt, system, max_tokens, temperature, decode):
    _, retries = _settings()
    for attempt in range(retries + 1):
        try:
            return decode(_request_once(prompt, system, max_tokens, temperature))
        except LLMError as exc:
            logger.warning('Model operation failure: code=%s attempt=%d', exc.code, attempt + 1)
            if not exc.retryable or attempt == retries:
                raise exc from None
            time.sleep(exc.delay)


def complete(prompt: str, system: str | None = None, max_tokens: int | None = None,
             temperature: float = 0.0) -> str:
    return _run(prompt, system, max_tokens, temperature, lambda text: text)


def validate_fields(result: dict, fields: dict[str, type]) -> None:
    """Exact types: e.g. the string 'false' must never become a passing True."""
    for name, expected_type in fields.items():
        if name not in result or type(result[name]) is not expected_type:
            raise LLMError('invalid_schema', 'The model returned an invalid structured response. Please try again.', retryable=True)
    if 'unsupported_claims' in fields and not all(isinstance(claim, str) for claim in result['unsupported_claims']):
        raise LLMError('invalid_schema', 'The model returned invalid claim details. Please try again.', retryable=True)


def complete_json(prompt: str, system: str | None = None, max_tokens: int | None = None,
                  validator: Callable[[dict], None] | None = None) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON keys')
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError('Non-finite JSON number')

    def decode(raw):
        # Accept one enclosing code fence, not arbitrary prose or partial JSON.
        match = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', raw.strip(), flags=re.DOTALL)
        cleaned = match.group(1) if match else raw
        try:
            result = json.loads(cleaned, object_pairs_hook=unique_object, parse_constant=reject_constant)
        except (json.JSONDecodeError, ValueError):
            raise LLMError('invalid_json', 'The model returned malformed JSON. Please try again.', retryable=True) from None
        if not isinstance(result, dict):
            raise LLMError('invalid_schema', 'The model response must be a JSON object. Please try again.', retryable=True)
        if validator:
            validator(result)
        return result
    hint = '\n\nRespond with ONLY one valid JSON object. No code fences, preamble or explanation.'
    return _run(prompt, (system or '') + hint, max_tokens, 0.0, decode)
