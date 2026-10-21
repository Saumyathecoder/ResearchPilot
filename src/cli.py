"""Command-line entry point for the agentic RAG system.

Usage:
    python -m src.cli "How does corrective RAG decide when to use web search?"
    python -m src.cli --interactive
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.agent.graph import run_agent
from src.config import CONFIG
from src.llm_client import LLMError

logging.basicConfig(
    level=logging.INFO if CONFIG.verbose else logging.WARNING,
    format="%(message)s",
)


def _print_result(question: str, state: dict) -> None:
    print("\n" + "=" * 80)
    print(f"Q: {question}")
    print("=" * 80)
    print("\n--- Agent trace ---")
    for line in state.get("trace", []):
        print(" ", line)
    print("\n--- Answer ---")
    print(state.get("answer", "(no answer produced)"))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start a REPL")
    args = parser.parse_args()

    if args.interactive or not args.question:
        print("Agentic RAG CLI. Type a question, or 'quit' to exit.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() in {"quit", "exit"}:
                break
            try:
                state = run_agent(question)
                _print_result(question, state)
            except LLMError as exc:
                print(f"Error: {exc}", file=sys.stderr)
    else:
        try:
            state = run_agent(args.question)
            _print_result(args.question, state)
        except LLMError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
