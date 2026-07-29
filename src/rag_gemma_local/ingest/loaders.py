from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".md", ".pdf", ".txt"}


def load_documents(raw_docs_path: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(raw_docs_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        else:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs
