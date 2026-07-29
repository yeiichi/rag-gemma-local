from __future__ import annotations

from rag_gemma_local.rag.service import answer_question


def main() -> None:
    questions = [
        "What model runtime is recommended?",
        "推奨するLLMランタイムは何ですか？",
    ]
    for question in questions:
        print(f"\nQuestion: {question}")
        print(answer_question(question, history=[]))


if __name__ == "__main__":
    main()
