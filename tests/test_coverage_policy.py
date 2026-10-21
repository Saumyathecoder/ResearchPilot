import pytest

import src.agent.nodes as nodes
from src.agent.graph import run_agent
from src.config import CONFIG
from src.graders.coverage_grader import _validate
from src.graders.generation_grader import GenerationVerdict
from src.graders.relevance_grader import GradedChunk
from src.llm_client import LLMError


class FixedStore:
    def similarity_search(self, query, **kwargs):
        return [{'chunk_id': str(i), 'text': f'Passage {i}', 'source_path': 'test.txt'} for i in range(4)]


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(CONFIG, 'enable_context_coverage', True)
    monkeypatch.setattr(CONFIG, 'max_query_rewrites', 2)
    monkeypatch.setattr(CONFIG, 'enable_web_fallback', True)
    monkeypatch.setattr('src.graders.relevance_grader.grade_chunk',
                        lambda q, c: GradedChunk(c['chunk_id'], c['text'], c, c['chunk_id'] == '0', 'mock'))
    monkeypatch.setattr(nodes, 'route_question', lambda q: {'route': 'retrieve', 'reason': 'mock'})
    monkeypatch.setattr(nodes, 'rewrite_query', lambda *args: 'rewritten')
    monkeypatch.setattr(nodes, 'generate_answer', lambda *args: 'Answer [1]')
    monkeypatch.setattr(nodes, 'grade_generation', lambda *args: GenerationVerdict(True, True, [], 'mock'))
    monkeypatch.setattr(nodes, 'web_search', lambda *args: [])


def test_complete_low_ratio_evidence_skips_recovery_but_is_still_graded(policy, monkeypatch):
    monkeypatch.setattr(nodes, 'check_context_coverage', lambda *args: {
        'sufficient': True, 'missing_aspects': [], 'reason': 'all parts supported'})
    monkeypatch.setattr(nodes, 'web_search', lambda *args: pytest.fail('Unnecessary web fallback'))
    state = run_agent('Question?', store=FixedStore())
    assert state['rewrite_count'] == 0
    assert not state['used_web_fallback']
    assert state['coverage_check_count'] == 1
    assert state['generation_verdict']['grounded']


@pytest.mark.parametrize('coverage', ['partial', 'failed', 'disabled'])
def test_incomplete_or_unavailable_check_retains_original_policy(policy, monkeypatch, coverage):
    def check(*args):
        if coverage == 'failed':
            raise LLMError('invalid_schema', 'Invalid coverage result')
        if coverage == 'disabled':
            pytest.fail('Disabled coverage checker was called')
        return {'sufficient': False, 'missing_aspects': ['second part'], 'reason': 'partial'}
    monkeypatch.setattr(CONFIG, 'enable_context_coverage', coverage != 'disabled')
    monkeypatch.setattr(nodes, 'check_context_coverage', check)
    state = run_agent('Multi-part question?', store=FixedStore())
    assert state['rewrite_count'] == 2
    assert state['used_web_fallback']
    assert state['coverage_check_count'] == (0 if coverage == 'disabled' else 3)
    if coverage == 'failed':
        assert state['context_coverage'] is None


@pytest.mark.parametrize('relevant', [True, False])
def test_high_ratio_or_no_relevant_evidence_does_not_add_check(policy, monkeypatch, relevant):
    monkeypatch.setattr('src.graders.relevance_grader.grade_chunk',
                        lambda q, c: GradedChunk(c['chunk_id'], c['text'], c, relevant, 'mock'))
    monkeypatch.setattr(nodes, 'check_context_coverage', lambda *args: pytest.fail('Unnecessary coverage check'))
    state = run_agent('Question?', store=FixedStore(), uploaded_corpus=True)
    assert state['coverage_check_count'] == 0
    assert not state['used_web_fallback']
    assert state['abstained'] == (not relevant)


@pytest.mark.parametrize('result', [
    {'sufficient': 'true', 'missing_aspects': [], 'reason': 'bad type'},
    {'sufficient': True, 'missing_aspects': ['missing part'], 'reason': 'contradictory'},
    {'sufficient': False, 'missing_aspects': [], 'reason': 'contradictory'},
    {'sufficient': False, 'missing_aspects': [3], 'reason': 'bad detail'},
])
def test_invalid_coverage_never_grants_sufficiency(result):
    with pytest.raises(LLMError):
        _validate(result)
