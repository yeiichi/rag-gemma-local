from __future__ import annotations

from rag_gemma_local.config import get_settings
from rag_gemma_local.index.chroma_store import get_retriever
from rag_gemma_local.llm.factory import get_llm
from rag_gemma_local.rag.formatting import format_chat_history, format_docs
from rag_gemma_local.rag.prompt_builder import (
    build_contextualize_prompt,
    build_qa_prompt,
)

_llm = None
_retriever = None


def _get_components():
    global _llm, _retriever
    settings = get_settings()
    if _llm is None:
        _llm = get_llm(settings)
    if _retriever is None:
        _retriever = get_retriever(settings)
    return settings, _llm, _retriever


def answer_question(question: str, history: list | None = None) -> str:
    settings, llm, retriever = _get_components()
    chat_history = format_chat_history(history or [])

    if chat_history:
        contextualize_prompt = build_contextualize_prompt(question, chat_history)
        search_query = llm.generate(contextualize_prompt).strip()
    else:
        search_query = question

    docs = retriever.invoke(search_query)
    context = format_docs(docs)
    if not context:
        return (
            f"No retrieved context was found in {settings.chroma_db_path}. "
            "Run ingestion first, or check RAW_DOCS_PATH and CHROMA_DB_PATH."
        )

    qa_prompt = build_qa_prompt(question, context, chat_history)
    return llm.generate(qa_prompt).strip()
