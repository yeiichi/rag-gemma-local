from __future__ import annotations

import shutil
from pathlib import Path

from rag_gemma_local.config import Settings, get_settings
from rag_gemma_local.index.chroma_store import get_vectorstore
from rag_gemma_local.ingest.chunking import split_documents
from rag_gemma_local.ingest.loaders import load_documents


def _is_safe_index_path(index_path: Path) -> bool:
    resolved = index_path.resolve()
    cwd = Path.cwd().resolve()
    return resolved != cwd and cwd in resolved.parents


def clear_index(settings: Settings) -> None:
    if not settings.chroma_db_path.exists():
        return
    if not settings.chroma_db_path.is_dir():
        raise ValueError(f"Chroma path is not a directory: {settings.chroma_db_path}")
    if not _is_safe_index_path(settings.chroma_db_path):
        raise ValueError(
            "Refusing to remove CHROMA_DB_PATH outside the project directory: "
            f"{settings.chroma_db_path}"
        )
    shutil.rmtree(settings.chroma_db_path)


def ingest(settings: Settings) -> None:
    docs = load_documents(settings.raw_docs_path)
    if not docs:
        print(f"No supported documents found under {settings.raw_docs_path}.")
        return

    chunks = split_documents(docs)
    vectorstore = get_vectorstore(settings)
    vectorstore.add_documents(chunks)
    print(
        f"Indexed {len(chunks)} chunks from {len(docs)} document pages/files "
        f"into {settings.chroma_db_path}."
    )


def reindex(settings: Settings) -> None:
    clear_index(settings)
    ingest(settings)


def main() -> None:
    ingest(get_settings())


if __name__ == "__main__":
    main()
