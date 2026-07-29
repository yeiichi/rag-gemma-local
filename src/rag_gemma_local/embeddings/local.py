from __future__ import annotations

import hashlib

from langchain_core.embeddings import Embeddings


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        normalized = text.lower()
        tokens = normalized.split()
        char_windows = [
            normalized[index : index + 3]
            for index in range(max(len(normalized) - 2, 0))
        ]

        for item in tokens + char_windows:
            digest = hashlib.blake2b(item.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self._dimensions
            vector[bucket] += 1.0

        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]
