from __future__ import annotations

from langchain_chroma import Chroma

from rag_gemma_local.config import Settings
from rag_gemma_local.embeddings.local import (
    HashEmbeddings,
    LocalSentenceTransformerEmbeddings,
)
from rag_gemma_local.index.retriever import SourceDiverseRetriever


def get_embeddings(settings: Settings):
    provider = settings.embedding_provider.lower().strip()
    if provider == "sentence_transformers":
        return LocalSentenceTransformerEmbeddings(settings.embedding_model)
    if provider == "hash":
        return HashEmbeddings()
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")


def get_vectorstore(settings: Settings) -> Chroma:
    return Chroma(
        persist_directory=str(settings.chroma_db_path),
        embedding_function=get_embeddings(settings),
    )


def get_retriever(settings: Settings):
    return SourceDiverseRetriever(
        vectorstore=get_vectorstore(settings),
        top_k=settings.rag_top_k,
        fetch_k=settings.rag_fetch_k,
        max_chunks_per_source=settings.rag_max_chunks_per_source,
    )
