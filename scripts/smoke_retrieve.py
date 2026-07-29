from __future__ import annotations

from rag_gemma_local.config import get_settings
from rag_gemma_local.index.chroma_store import get_retriever


def main() -> None:
    settings = get_settings()
    retriever = get_retriever(settings)
    questions = [
        "What model runtime is recommended?",
        "推奨するLLMランタイムは何ですか？",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        docs = retriever.invoke(question)
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content.replace("\n", " ")[:160]
            print(f"{index}. {source}: {preview}")


if __name__ == "__main__":
    main()
