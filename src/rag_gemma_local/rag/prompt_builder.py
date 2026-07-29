from __future__ import annotations

import re

from rag_gemma_local.rag.prompts import CONTEXTUALIZE_PROMPT, QA_SYSTEM_PROMPT


def answer_language_instruction(question: str) -> str:
    has_japanese = any(
        "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff"
        for char in question
    )
    if has_japanese:
        return "Answer in Japanese."
    return "Answer in English."


def build_contextualize_prompt(question: str, chat_history: str) -> str:
    if not chat_history:
        return question
    return (
        f"{CONTEXTUALIZE_PROMPT}\n\n"
        f"Chat history:\n{chat_history}\n\n"
        f"Latest question:\n{question}\n\n"
        "Standalone query:"
    )


def build_qa_prompt(question: str, context: str, chat_history: str = "") -> str:
    history_section = f"\nChat history:\n{chat_history}\n" if chat_history else ""
    language_instruction = answer_language_instruction(question)
    coverage_instruction = build_source_coverage_instruction(context)
    return (
        f"{QA_SYSTEM_PROMPT.format(context=context)}"
        f"\n{language_instruction}\n"
        f"{coverage_instruction}\n"
        f"{history_section}\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def build_source_coverage_instruction(context: str) -> str:
    source_labels = sorted(set(re.findall(r"Content from \[([^\]]+)\]", context)))
    if len(source_labels) <= 1:
        return "Use the retrieved source if it is relevant."

    sources = ", ".join(source_labels)
    return (
        "Coverage requirement: The retrieved context contains multiple source "
        f"files ({sources}). If the question asks about multiple documents or "
        "the corpus as a whole, include at least one distinct point from each "
        "source file and cite each source label. Use a source-by-source answer "
        "structure when helpful. Do not answer from only one source file unless "
        "the other retrieved source files are clearly irrelevant."
    )
