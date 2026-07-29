from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document


class VectorStore(Protocol):
    def similarity_search(self, query: str, k: int) -> list[Document]:
        """Return documents ranked by vector similarity."""


def source_key(doc: Document) -> str:
    source = doc.metadata.get("source", "Unknown Document")
    return Path(str(source)).name


def select_source_diverse_docs(
    docs: list[Document],
    *,
    top_k: int,
    max_chunks_per_source: int,
) -> list[Document]:
    if top_k <= 0:
        return []

    selected: list[Document] = []
    selected_ids: set[int] = set()
    source_counts: Counter[str] = Counter()

    for index, doc in enumerate(docs):
        key = source_key(doc)
        if source_counts[key] > 0:
            continue
        selected.append(doc)
        selected_ids.add(index)
        source_counts[key] += 1
        if len(selected) >= top_k:
            return selected

    for index, doc in enumerate(docs):
        if index in selected_ids:
            continue
        key = source_key(doc)
        if max_chunks_per_source > 0 and source_counts[key] >= max_chunks_per_source:
            continue
        selected.append(doc)
        source_counts[key] += 1
        if len(selected) >= top_k:
            break

    return selected


class SourceDiverseRetriever:
    def __init__(
        self,
        *,
        vectorstore: VectorStore,
        top_k: int,
        fetch_k: int,
        max_chunks_per_source: int,
    ) -> None:
        self._vectorstore = vectorstore
        self._top_k = top_k
        self._fetch_k = max(fetch_k, top_k)
        self._max_chunks_per_source = max_chunks_per_source

    def invoke(self, query: str) -> list[Document]:
        candidates = self._vectorstore.similarity_search(query, k=self._fetch_k)
        return select_source_diverse_docs(
            candidates,
            top_k=self._top_k,
            max_chunks_per_source=self._max_chunks_per_source,
        )
