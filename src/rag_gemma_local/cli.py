from __future__ import annotations

import argparse

from rag_gemma_local.rag.service import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local Gemma RAG.")
    parser.add_argument("question", help="Question to ask the local document index.")
    args = parser.parse_args()
    print(answer_question(args.question, history=[]))


if __name__ == "__main__":
    main()
