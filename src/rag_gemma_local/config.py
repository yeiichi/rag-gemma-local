from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    chroma_db_path: Path
    raw_docs_path: Path
    rag_top_k: int
    rag_fetch_k: int
    rag_max_chunks_per_source: int
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    gemma_gguf_path: Path
    llama_cpp_n_ctx: int
    llama_cpp_n_threads: int
    llama_cpp_max_tokens: int
    llama_cpp_repeat_penalty: float
    llama_cpp_force_cpu_devices: bool
    hf_model_repo_id: str
    hf_model_filename: str
    ollama_base_url: str
    ollama_model: str


def get_settings() -> Settings:
    return Settings(
        chroma_db_path=Path(os.environ.get("CHROMA_DB_PATH", "./data/indexes/chroma")),
        raw_docs_path=Path(os.environ.get("RAW_DOCS_PATH", "./data/raw")),
        rag_top_k=_int_env("RAG_TOP_K", 5),
        rag_fetch_k=_int_env("RAG_FETCH_K", 20),
        rag_max_chunks_per_source=_int_env("RAG_MAX_CHUNKS_PER_SOURCE", 2),
        embedding_provider=os.environ.get(
            "EMBEDDING_PROVIDER",
            "sentence_transformers",
        ),
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL",
            "intfloat/multilingual-e5-small",
        ),
        llm_provider=os.environ.get("LLM_PROVIDER", "llama_cpp"),
        gemma_gguf_path=Path(
            os.environ.get("GEMMA_GGUF_PATH", "./models/llm/gemma-2b-it-q4.gguf")
        ),
        llama_cpp_n_ctx=_int_env("LLAMA_CPP_N_CTX", 4096),
        llama_cpp_n_threads=_int_env("LLAMA_CPP_N_THREADS", 4),
        llama_cpp_max_tokens=_int_env("LLAMA_CPP_MAX_TOKENS", 512),
        llama_cpp_repeat_penalty=float(os.environ.get("LLAMA_CPP_REPEAT_PENALTY", 1.1)),
        llama_cpp_force_cpu_devices=_bool_env("LLAMA_CPP_FORCE_CPU_DEVICES", True),
        hf_model_repo_id=os.environ.get(
            "HF_MODEL_REPO_ID",
            "bartowski/gemma-2-2b-it-GGUF",
        ),
        hf_model_filename=os.environ.get(
            "HF_MODEL_FILENAME",
            "gemma-2-2b-it-Q4_K_M.gguf",
        ),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "gemma:2b"),
    )
