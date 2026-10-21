import sys
from types import SimpleNamespace

from src.tools.web_search import web_search


def fake_search(monkeypatch, rows=None, error=None):
    class Search:
        def __init__(self, **kwargs):
            assert kwargs['timeout'] == 5

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def text(self, query, **kwargs):
            if error:
                raise error
            return rows
    monkeypatch.setitem(sys.modules, 'ddgs', SimpleNamespace(DDGS=Search))


def test_web_sources_require_url_and_nonempty_body(monkeypatch):
    fake_search(monkeypatch, [None, {}, {'href': '', 'body': 'text'},
                             {'href': 'javascript:alert(1)', 'body': 'text'},
                             {'href': 'https://example.org', 'body': '  '},
                             {'href': 'https://user:password@example.org', 'body': 'text'},
                             {'href': 'https://example.org', 'body': 'Useful text', 'title': 'Title'},
                             {'href': 'https://example.org', 'body': 'Duplicate'}])
    results = web_search('question')
    assert len(results) == 1
    assert results[0]['source_path'] == 'https://example.org'
    assert results[0]['text'] == 'Title\nUseful text'
    assert web_search('question')[0]['chunk_id'] == results[0]['chunk_id']


def test_search_failure_does_not_log_exception_contents(monkeypatch, caplog):
    fake_search(monkeypatch, error=RuntimeError('private-error-detail'))
    assert web_search('question') == []
    assert 'private-error-detail' not in caplog.text
