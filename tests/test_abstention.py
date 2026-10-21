import pytest

import src.agent.nodes as nodes
from src.agent.graph import run_agent
from src.config import CONFIG
from src.graders.generation_grader import GenerationVerdict
from src.graders.relevance_grader import GradedChunk


class Store:
    def similarity_search(self, query, **kwargs):
        return [{'chunk_id': 'one', 'text': 'Amber rocket launch.', 'source_path': 'a.txt'}]


def prepare(monkeypatch, *, web_enabled=True):
    monkeypatch.setattr(CONFIG, 'max_query_rewrites', 0)
    monkeypatch.setattr(CONFIG, 'enable_web_fallback', web_enabled)
    monkeypatch.setattr(nodes, 'route_question', lambda q: {'route': 'retrieve', 'reason': 'test'})
    monkeypatch.setattr('src.graders.relevance_grader.grade_chunk',
                        lambda q, c: GradedChunk(c['chunk_id'], c['text'], c, False, 'irrelevant'))


@pytest.mark.parametrize('web_enabled', [True, False])
def test_sample_empty_evidence_skips_generation_and_judges(monkeypatch, web_enabled):
    prepare(monkeypatch, web_enabled=web_enabled)
    monkeypatch.setattr(nodes, 'web_search', lambda q: [])
    monkeypatch.setattr(nodes, 'generate_answer', lambda *args: pytest.fail('No generation with empty evidence'))
    monkeypatch.setattr(nodes, 'grade_generation', lambda *args: pytest.fail('No judge calls for abstention'))
    state = run_agent('Missing fact?', store=Store())
    assert state['abstained']
    assert state['generation_retry_count'] == 0
    assert 'generation_verdict' not in state
    assert state['used_web_fallback'] == web_enabled
    assert 'Web search is disabled for uploads' not in state['answer']
    assert ('web search did not provide' in state['answer']) == web_enabled


def test_nonempty_web_evidence_is_still_generated_and_graded(monkeypatch):
    prepare(monkeypatch)
    monkeypatch.setattr(nodes, 'web_search', lambda q: [{'text': 'Supported fact', 'source_path': 'https://example.org'}])
    calls = []
    def generate(q, chunks):
        calls.append('generate')
        assert chunks[0]['text'] == 'Supported fact'
        return 'Supported answer [1]'
    def grade(*args):
        calls.append('grade')
        return GenerationVerdict(True, True, [], 'test')
    monkeypatch.setattr(nodes, 'generate_answer', generate)
    monkeypatch.setattr(nodes, 'grade_generation', grade)
    state = run_agent('Question?', store=Store())
    assert calls == ['generate', 'grade']
    assert not state['abstained']
    assert state['answer'] == 'Supported answer [1]'


def test_empty_rewrite_restores_original_intent(monkeypatch):
    import src.graders.query_rewriter as rewriter
    monkeypatch.setattr(rewriter, 'complete', lambda *args, **kwargs: '  ')
    assert rewriter.rewrite_query('How do I make tiramisu?', 'RAG architecture') == 'How do I make tiramisu?'
