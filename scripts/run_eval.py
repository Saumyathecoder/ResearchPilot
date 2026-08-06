"""Run the evaluation harness: naive RAG vs agentic self-correcting RAG,
over data/eval_dataset.json. Writes eval_results/results.json and
eval_results/results.md.

Requires ANTHROPIC_API_KEY to be set (every metric here uses an LLM judge
or the LLMs inside each pipeline).

Usage:
    python -m scripts.run_eval
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from src.agent.graph import run_agent
from src.evaluation.metrics import (
    faithfulness_and_relevance,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)
from src.evaluation.naive_baseline import naive_rag_answer

DATASET_PATH = Path("data/eval_dataset.json")
RESULTS_DIR = Path("eval_results")


def _eval_naive(item: dict) -> dict:
    result = naive_rag_answer(item["question"])
    quality = faithfulness_and_relevance(item["question"], result["retrieved"], result["answer"])
    return {
        "precision": retrieval_precision_at_k(result["retrieved"], item["expected_doc_ids"]),
        "recall": retrieval_recall_at_k(result["retrieved"], item["expected_doc_ids"]),
        **quality,
        "answer": result["answer"],
    }


def _eval_agentic(item: dict) -> dict:
    state = run_agent(item["question"])
    chunks = state.get("relevant_chunks", [])
    answer = state.get("answer", "")
    quality = faithfulness_and_relevance(item["question"], chunks, answer)
    return {
        "precision": retrieval_precision_at_k(chunks, item["expected_doc_ids"]),
        "recall": retrieval_recall_at_k(chunks, item["expected_doc_ids"]),
        **quality,
        "answer": answer,
        "used_web_fallback": state.get("used_web_fallback", False),
        "rewrite_count": state.get("rewrite_count", 0),
        "generation_retry_count": state.get("generation_retry_count", 0),
        "route": state.get("route"),
    }


def _avg(values: list[float]) -> float:
    return round(statistics.mean(values), 3) if values else 0.0


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text())
    RESULTS_DIR.mkdir(exist_ok=True)

    per_question = []
    for item in dataset:
        print(f"Evaluating: {item['question'][:70]}...")
        naive = _eval_naive(item)
        agentic = _eval_agentic(item)
        per_question.append({"question": item["question"], "naive": naive, "agentic": agentic})

    summary = {
        "naive": {
            "avg_precision": _avg([r["naive"]["precision"] for r in per_question]),
            "avg_recall": _avg([r["naive"]["recall"] for r in per_question]),
            "grounded_rate": _avg([float(r["naive"]["grounded"]) for r in per_question]),
            "relevance_rate": _avg([float(r["naive"]["addresses_question"]) for r in per_question]),
            "pass_rate": _avg([float(r["naive"]["passed"]) for r in per_question]),
        },
        "agentic": {
            "avg_precision": _avg([r["agentic"]["precision"] for r in per_question]),
            "avg_recall": _avg([r["agentic"]["recall"] for r in per_question]),
            "grounded_rate": _avg([float(r["agentic"]["grounded"]) for r in per_question]),
            "relevance_rate": _avg([float(r["agentic"]["addresses_question"]) for r in per_question]),
            "pass_rate": _avg([float(r["agentic"]["passed"]) for r in per_question]),
        },
    }

    (RESULTS_DIR / "results.json").write_text(
        json.dumps({"summary": summary, "per_question": per_question}, indent=2)
    )

    md = ["# Evaluation Results: Naive RAG vs Agentic Self-Correcting RAG", ""]
    md.append(f"Evaluated on {len(dataset)} questions from `data/eval_dataset.json`.\n")
    md.append("| Metric | Naive RAG | Agentic RAG |")
    md.append("|---|---|---|")
    for key, label in [
        ("avg_precision", "Retrieval precision@k"),
        ("avg_recall", "Retrieval recall@k"),
        ("grounded_rate", "Groundedness (faithfulness) rate"),
        ("relevance_rate", "Answer-relevance rate"),
        ("pass_rate", "Overall pass rate (grounded AND relevant)"),
    ]:
        md.append(f"| {label} | {summary['naive'][key]} | {summary['agentic'][key]} |")
    md.append("")
    md.append(
        "Full per-question results (including generated answers and agent traces) "
        "are in `eval_results/results.json`."
    )
    (RESULTS_DIR / "results.md").write_text("\n".join(md))

    print("\n" + "\n".join(md))
    print("\nWrote eval_results/results.json and eval_results/results.md")


if __name__ == "__main__":
    main()
