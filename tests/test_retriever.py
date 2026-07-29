from langchain_core.documents import Document

from rag_gemma_local.index.retriever import select_source_diverse_docs


def _doc(source: str, content: str = "content") -> Document:
    return Document(page_content=content, metadata={"source": source})


def test_select_source_diverse_docs_prefers_distinct_sources() -> None:
    docs = [
        _doc("english_policy.md", "en 1"),
        _doc("english_policy.md", "en 2"),
        _doc("japanese_policy.md", "ja 1"),
        _doc("english_policy.md", "en 3"),
    ]

    selected = select_source_diverse_docs(
        docs,
        top_k=3,
        max_chunks_per_source=2,
    )

    assert [doc.metadata["source"] for doc in selected] == [
        "english_policy.md",
        "japanese_policy.md",
        "english_policy.md",
    ]


def test_select_source_diverse_docs_respects_max_chunks_per_source() -> None:
    docs = [
        _doc("english_policy.md", "en 1"),
        _doc("japanese_policy.md", "ja 1"),
        _doc("english_policy.md", "en 2"),
        _doc("japanese_policy.md", "ja 2"),
        _doc("english_policy.md", "en 3"),
    ]

    selected = select_source_diverse_docs(
        docs,
        top_k=5,
        max_chunks_per_source=2,
    )

    assert [doc.page_content for doc in selected] == ["en 1", "ja 1", "en 2", "ja 2"]
