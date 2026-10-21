"""Streamlit wiring checks. Real browser/provider verification remains manual."""
from streamlit.testing.v1 import AppTest
from pathlib import Path

from src.ingestion.uploads import UploadCorpus

APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_upload_mode_requires_index_and_clear_disables_ask():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.radio[0].set_value('My uploads').run()
    assert not app.exception
    assert button(app, 'Ask').disabled
    app.session_state['upload_corpus'] = UploadCorpus([('Mission.txt', b'Amber rocket launch.')])
    app.run()
    assert not button(app, 'Ask').disabled
    button(app, 'Clear uploaded corpus').click().run()
    assert button(app, 'Ask').disabled


def test_upload_ask_uses_session_store_and_displays_page(monkeypatch):
    import src.agent.graph as graph
    corpus = UploadCorpus([('Mission.txt', b'Amber rocket launch.')])
    def answer(question, **kwargs):
        assert kwargs['store'] is corpus
        assert kwargs['uploaded_corpus'] is True
        return {'answer': 'Amber [1]', 'relevant_chunks': [
            {'source_path': 'Mission.pdf', 'page_number': 2, 'text': 'Amber rocket launch.'}]}
    monkeypatch.setattr(graph, 'run_agent', answer)
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.session_state['upload_corpus'] = corpus
    app.radio[0].set_value('My uploads').run()
    app.text_input[0].set_value('Which rocket?')
    button(app, 'Ask').click().run()
    assert not app.exception
    assert any('Mission.pdf — page 2' in t.value for t in app.text)


def test_api_failure_has_safe_ui_message(monkeypatch):
    import src.agent.graph as graph
    def fail(*args, **kwargs):
        raise RuntimeError('sensitive-provider-error')
    monkeypatch.setattr(graph, 'run_agent', fail)
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.text_input[0].set_value('Which rocket?')
    button(app, 'Ask').click().run()
    assert not app.exception
    assert any('could not be completed' in error.value for error in app.error)
    assert all('sensitive-provider-error' not in error.value for error in app.error)


def test_abstention_ui_does_not_display_passing_quality_metrics(monkeypatch):
    import src.agent.graph as graph
    monkeypatch.setattr(graph, 'run_agent', lambda *args, **kwargs: {
        'answer': 'No relevant context.', 'route': 'retrieve', 'abstained': True})
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.text_input[0].set_value('Missing fact?')
    button(app, 'Ask').click().run()
    assert not app.exception
    assert [m.label for m in app.metric] == ['Evidence']
    assert app.metric[0].value == 'Not found'
    assert any('grading were skipped' in item.value for item in app.info)


def test_typed_model_failure_displays_actionable_error_without_answer(monkeypatch):
    import src.agent.graph as graph
    from src.llm_client import LLMError
    def fail(*args, **kwargs):
        raise LLMError('rate_limit', 'The model service rate limit was reached. Please wait before trying again.')
    monkeypatch.setattr(graph, 'run_agent', fail)
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.text_input[0].set_value('A question')
    button(app, 'Ask').click().run()
    assert not app.exception
    assert any('rate limit' in error.value for error in app.error)
    assert not app.metric
    assert all(heading.value != 'Answer' for heading in app.subheader)
