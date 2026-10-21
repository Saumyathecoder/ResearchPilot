import numpy as np
import pytest

from src.evaluation.retrieval_benchmark import evaluate_retriever
from src.ingestion.chunker import Chunk


class FixedEmbedder:
    def fit(self, texts):
        pass

    def embed(self, texts):
        return np.array([{'first': [1., 0.], 'second': [0., 1.],
                          'question': [0., 2.], 'unanswerable': [0., 0.]}[t] for t in texts])


def corpus():
    return [Chunk('a::0', 'a', 'a.txt', 'first', ''),
            Chunk('b::0', 'b', 'b.txt', 'second', '')]


def test_known_rank_and_out_of_scope_exclusion():
    result = evaluate_retriever(corpus(), [
        {'question': 'question', 'expected_doc_ids': ['a']},
        {'question': 'unanswerable', 'expected_doc_ids': []}], FixedEmbedder(), k=2)
    assert result['summary'] == {'mean_precision': .5, 'mean_recall': 1., 'mean_reciprocal_rank': .5}
    assert result['in_scope_questions'] == result['out_of_scope_questions'] == 1
    assert result['per_question'][0]['hits'][0]['doc_id'] == 'b'
    assert 'precision' not in result['per_question'][1]


def test_k_and_tie_order():
    result = evaluate_retriever(corpus(), [{'question': 'unanswerable', 'expected_doc_ids': []}], FixedEmbedder(), k=10)
    assert [h['doc_id'] for h in result['per_question'][0]['hits']] == ['a', 'b']


@pytest.mark.parametrize('expected', [['missing'], ['a', 'a'], 'a'])
def test_invalid_labels(expected):
    with pytest.raises(ValueError):
        evaluate_retriever(corpus(), [{'question': 'question', 'expected_doc_ids': expected}], FixedEmbedder())


def test_empty_input_and_zero_k():
    with pytest.raises(ValueError):
        evaluate_retriever([], [], FixedEmbedder())
    with pytest.raises(ValueError):
        evaluate_retriever(corpus(), [], FixedEmbedder(), k=0)


def test_loaded_model_provenance_is_optional_and_digest_only():
    from types import SimpleNamespace as NS
    from scripts.benchmark_retrieval import model_revision
    assert model_revision(FixedEmbedder()) is None
    model = NS(modules=lambda: [NS(config=NS(_commit_hash='a' * 40))])
    assert model_revision(NS(model=model)) == 'a' * 40
    model = NS(modules=lambda: [NS(config=NS(_commit_hash='arbitrary-text'))])
    assert model_revision(NS(model=model)) is None
