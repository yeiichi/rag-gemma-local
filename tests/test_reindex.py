from pathlib import Path

import pytest

from rag_gemma_local.config import Settings
from rag_gemma_local.ingest.pipeline import clear_index


def _settings(chroma_db_path: Path) -> Settings:
    return Settings(
        chroma_db_path=chroma_db_path,
        raw_docs_path=Path("data/raw"),
        rag_top_k=5,
        rag_fetch_k=20,
        rag_max_chunks_per_source=2,
        embedding_provider="hash",
        embedding_model="test",
        llm_provider="context",
        gemma_gguf_path=Path("models/llm/model.gguf"),
        llama_cpp_n_ctx=512,
        llama_cpp_n_threads=4,
        llama_cpp_max_tokens=64,
        llama_cpp_repeat_penalty=1.1,
        llama_cpp_force_cpu_devices=True,
        hf_model_repo_id="repo/model",
        hf_model_filename="model.gguf",
        ollama_base_url="http://localhost:11434",
        ollama_model="gemma:2b",
    )


def test_clear_index_removes_project_index(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    index_path = tmp_path / "data" / "indexes" / "test"
    index_path.mkdir(parents=True)
    (index_path / "chroma.sqlite3").write_text("db", encoding="utf-8")

    clear_index(_settings(index_path))

    assert not index_path.exists()


def test_clear_index_refuses_outside_project_path(tmp_path, monkeypatch) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()
    monkeypatch.chdir(project_path)
    outside_path = tmp_path / "outside-index"
    outside_path.mkdir()

    with pytest.raises(ValueError, match="outside the project directory"):
        clear_index(_settings(outside_path))
