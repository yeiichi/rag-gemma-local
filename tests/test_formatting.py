from langchain_core.documents import Document

from rag_gemma_local.rag.formatting import format_docs


def test_format_docs_includes_source_inventory() -> None:
    context = format_docs(
        [
            Document(page_content="English notes", metadata={"source": "english.md"}),
            Document(page_content="Japanese notes", metadata={"source": "japanese.md"}),
        ]
    )

    assert "Retrieved source files:" in context
    assert "- english.md" in context
    assert "- japanese.md" in context
    assert "Retrieved context blocks:" in context
