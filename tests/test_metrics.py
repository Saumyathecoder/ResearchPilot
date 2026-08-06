from src.evaluation.metrics import retrieval_precision_at_k, retrieval_recall_at_k


def test_precision_all_relevant():
    chunks = [{"doc_id": "a"}, {"doc_id": "a"}, {"doc_id": "b"}]
    assert retrieval_precision_at_k(chunks, ["a", "b"]) == 1.0


def test_precision_partial_relevant():
    chunks = [{"doc_id": "a"}, {"doc_id": "c"}]
    assert retrieval_precision_at_k(chunks, ["a"]) == 0.5


def test_precision_out_of_scope_question_no_retrieval_is_perfect():
    assert retrieval_precision_at_k([], []) == 1.0


def test_precision_out_of_scope_question_but_something_retrieved_is_zero():
    assert retrieval_precision_at_k([{"doc_id": "a"}], []) == 0.0


def test_recall_finds_all_expected_docs():
    chunks = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]
    assert retrieval_recall_at_k(chunks, ["a", "b"]) == 1.0


def test_recall_missing_expected_doc():
    chunks = [{"doc_id": "a"}]
    assert retrieval_recall_at_k(chunks, ["a", "b"]) == 0.5
