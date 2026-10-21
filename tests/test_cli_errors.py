from src import cli
from src.llm_client import LLMError


def fail(*args, **kwargs):
    raise LLMError('timeout', 'The model request timed out. Please try again.')


def test_cli_model_failure_returns_nonzero_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['cli', 'a question'])
    monkeypatch.setattr(cli, 'run_agent', fail)
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert 'timed out' in captured.err
    assert 'Traceback' not in captured.err
    assert '--- Answer ---' not in captured.out


def test_interactive_cli_recovers_after_model_failure(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['cli', '--interactive'])
    monkeypatch.setattr(cli, 'run_agent', fail)
    answers = iter(['a question', 'quit'])
    monkeypatch.setattr('builtins.input', lambda *args: next(answers))
    assert cli.main() == 0
    assert 'timed out' in capsys.readouterr().err
