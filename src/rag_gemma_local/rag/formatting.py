from __future__ import annotations

import os
from collections.abc import Iterable


def format_docs(docs: Iterable) -> str:
    formatted = []
    labels = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Unknown Document"))
        page = doc.metadata.get("page")
        page_label = f" - Page {page + 1}" if isinstance(page, int) else ""
        label = f"{source}{page_label}"
        if label not in labels:
            labels.append(label)
        formatted.append(f"Content from [{label}]:\n{doc.page_content}")

    if not formatted:
        return ""

    source_inventory = "\n".join(f"- {label}" for label in labels)
    context_blocks = "\n\n".join(formatted)
    return (
        "Retrieved source files:\n"
        f"{source_inventory}\n\n"
        "Retrieved context blocks:\n"
        f"{context_blocks}"
    )


def format_chat_history(history: list) -> str:
    lines = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                lines.append(f"{role}: {content}")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lines.append(f"user: {item[0]}")
            lines.append(f"assistant: {item[1]}")
    return "\n".join(lines)
