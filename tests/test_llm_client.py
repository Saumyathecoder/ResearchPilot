"""Provider failures and malformed model responses, without network or real keys."""
import json
from types import SimpleNamespace as NS

import groq
import httpx
import pytest

import src.llm_client as llm
from src.config import CONFIG


def response(text='answer', reason='stop'):
    return NS(choices=[NS(message=NS(content=text), finish_reason=reason)])


@pytest.fixture
def setup_client(monkeypatch):
    monkeypatch.setattr(CONFIG, 'llm_provider', 'groq')
    monkeypatch.setattr(CONFIG, 'llm_model', 'openai/gpt-oss-120b')
    monkeypatch.setattr(CONFIG, 'llm_timeout_seconds', 30.)
    monkeypatch.setattr(CONFIG, 'llm_max_retries', 1)
    sleeps = []
    monkeypatch.setattr(llm.time, 'sleep', sleeps.append)
    def setup(items):
        calls = []
        items = iter(items)
        def create(**kwargs):
            calls.append(kwargs)
            item = next(items)
            if isinstance(item, Exception):
                raise item
            return item
        monkeypatch.setattr(llm, '_client', NS(chat=NS(completions=NS(create=create))))
        return calls, sleeps
    return setup


def status_error(status, headers=None):
    request = httpx.Request('POST', 'https://example.org')
    return groq.APIStatusError('private-provider-body', response=httpx.Response(status, request=request, headers=headers), body={'secret': 'test-only-secret'})


def test_gpt_oss_options_preserved(setup_client):
    calls, _ = setup_client([response()])
    assert llm.complete('question', max_tokens=100) == 'answer'
    assert calls[0]['max_completion_tokens'] == 4096
    assert calls[0]['reasoning_effort'] == 'low'


def test_other_groq_model_does_not_get_oss_options(setup_client, monkeypatch):
    calls, _ = setup_client([response()])
    monkeypatch.setattr(CONFIG, 'llm_model', 'another-model')
    llm.complete('question', max_tokens=100)
    assert calls[0]['max_completion_tokens'] == 100
    assert 'reasoning_effort' not in calls[0]


@pytest.mark.parametrize('error', [
    status_error(500), status_error(408), status_error(409),
    status_error(429, {'retry-after': '0'}),
    groq.APITimeoutError(request=httpx.Request('POST', 'https://example.org')),
    groq.APIConnectionError(request=httpx.Request('POST', 'https://example.org')),
])
def test_transient_error_retries_once_then_recovers(setup_client, error):
    calls, sleeps = setup_client([error, response()])
    assert llm.complete('question') == 'answer'
    assert len(calls) == 2 and len(sleeps) == 1


@pytest.mark.parametrize('status,headers', [(401, None), (403, None), (400, None),
                                        (429, {'retry-after': '60'}),
                                        (429, {'retry-after': 'not-a-number'})])
def test_permanent_errors_and_long_rate_limits_not_retried(setup_client, status, headers, caplog):
    calls, sleeps = setup_client([status_error(status, headers)])
    with pytest.raises(llm.LLMError) as error:
        llm.complete('private prompt')
    assert len(calls) == 1 and not sleeps
    assert 'private-provider-body' not in str(error.value) + caplog.text
    assert 'test-only-secret' not in str(error.value) + caplog.text
    assert 'private prompt' not in caplog.text


def test_transient_retry_budget_exhausted(setup_client):
    calls, sleeps = setup_client([status_error(503), status_error(503)])
    with pytest.raises(llm.LLMError, match='temporarily unavailable'):
        llm.complete('question')
    assert len(calls) == 2 and len(sleeps) == 1


def test_empty_output_recovery(setup_client):
    calls, _ = setup_client([response('  '), response('valid')])
    assert llm.complete('question') == 'valid'
    assert len(calls) == 2


def test_empty_output_exhausted(setup_client):
    calls, _ = setup_client([response(None), response('')])
    with pytest.raises(llm.LLMError, match='empty response'):
        llm.complete('question')
    assert len(calls) == 2


def test_truncated_response_not_returned_or_retried(setup_client, caplog):
    calls, sleeps = setup_client([response('private-partial-output', 'length')])
    with pytest.raises(llm.LLMError, match='token limit'):
        llm.complete('question')
    assert len(calls) == 1 and not sleeps
    assert 'private-partial-output' not in caplog.text


def test_json_and_transport_share_one_budget(setup_client):
    calls, sleeps = setup_client([status_error(500), response('private-invalid-json')])
    with pytest.raises(llm.LLMError, match='malformed JSON'):
        llm.complete_json('question')
    assert len(calls) == 2 and len(sleeps) == 1


@pytest.mark.parametrize('bad', ['private-invalid-json', '[]', 'null', '{"relevant":"false","reason":"test"}', '{}'])
def test_invalid_json_or_schema_can_recover(setup_client, bad):
    calls, _ = setup_client([response(bad), response('{"relevant":false,"reason":"test"}')])
    result = llm.complete_json('question', validator=lambda r: llm.validate_fields(r, {'relevant': bool, 'reason': str}))
    assert result['relevant'] is False
    assert len(calls) == 2


def test_bad_json_never_leaks_raw_output(setup_client, caplog):
    setup_client([response('private-invalid-json'), response('private-invalid-json')])
    with pytest.raises(llm.LLMError) as error:
        llm.complete_json('question')
    assert 'private-invalid-json' not in str(error.value) + caplog.text


def test_code_fence_is_supported(setup_client):
    setup_client([response('```json\n{"ok":true}\n```')])
    assert llm.complete_json('question') == {'ok': True}


@pytest.mark.parametrize('raw', ['{"ok":true,"ok":false}', '{"value":NaN}', '{"value":Infinity}'])
def test_nonstandard_json_is_rejected(setup_client, raw):
    setup_client([response(raw), response(raw)])
    with pytest.raises(llm.LLMError, match='malformed JSON'):
        llm.complete_json('question')


def test_zero_retries_is_respected(setup_client, monkeypatch):
    calls, sleeps = setup_client([response('')])
    monkeypatch.setattr(CONFIG, 'llm_max_retries', 0)
    with pytest.raises(llm.LLMError):
        llm.complete('question')
    assert len(calls) == 1 and not sleeps


@pytest.mark.parametrize('timeout,retries', [(0, 1), (float('inf'), 1), (30, 3)])
def test_unsafe_request_settings_are_rejected(monkeypatch, timeout, retries):
    monkeypatch.setattr(CONFIG, 'llm_timeout_seconds', timeout)
    monkeypatch.setattr(CONFIG, 'llm_max_retries', retries)
    with pytest.raises(llm.LLMError, match='configuration'):
        llm.complete('question')


def test_relevance_string_false_is_rejected_by_real_grader(setup_client):
    from src.graders.relevance_grader import grade_chunk
    setup_client([response('{"relevant":"false","reason":"test"}') for _ in range(2)])
    with pytest.raises(llm.LLMError):
        grade_chunk('question', {'chunk_id': 'one', 'text': 'passage'})


def test_invalid_route_is_not_silently_replaced(setup_client):
    from src.graders.router import route_question
    setup_client([response('{"route":"unsupported","reason":"test"}') for _ in range(2)])
    with pytest.raises(llm.LLMError):
        route_question('question')


@pytest.mark.parametrize('claims', ['invalid-list', [3], ['unsupported claim']])
def test_invalid_grounding_details_never_pass(setup_client, claims):
    from src.graders.generation_grader import grade_generation
    raw = json.dumps({'grounded': True, 'unsupported_claims': claims, 'reason': 'test'})
    setup_client([response(raw), response(raw)])
    with pytest.raises(llm.LLMError):
        grade_generation('question', [{'text': 'passage'}], 'answer')


def test_constructor_has_timeout_without_sdk_retries(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm, '_client', None)
    monkeypatch.setattr(CONFIG, 'llm_provider', 'groq')
    monkeypatch.setattr(CONFIG, 'groq_api_key', 'test-only-placeholder')
    monkeypatch.setattr(CONFIG, 'llm_timeout_seconds', 17.)
    def constructor(**kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setattr(groq, 'Groq', constructor)
    llm._get_client()
    assert captured['timeout'] == 17 and captured['max_retries'] == 0


def test_anthropic_text_and_truncation(monkeypatch, setup_client):
    setup_client([])
    monkeypatch.setattr(CONFIG, 'llm_provider', 'anthropic')
    def create(**kwargs):
        return NS(stop_reason='end_turn', content=[NS(type='text', text='supported answer')])
    monkeypatch.setattr(llm, '_client', NS(messages=NS(create=create)))
    assert llm.complete('question') == 'supported answer'
    monkeypatch.setattr(llm._client.messages, 'create', lambda **kwargs: NS(stop_reason='max_tokens', content=[]))
    with pytest.raises(llm.LLMError, match='token limit'):
        llm.complete('question')
