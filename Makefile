.PHONY: help sync lint test run ingest reindex smoke-hash smoke-retrieve smoke-answer download-model check-llama-cpp docs clean

help:
	@printf "Targets:\n"
	@printf "  sync    Install dependencies with uv\n"
	@printf "  lint    Run ruff checks\n"
	@printf "  test    Run tests\n"
	@printf "  run     Launch the Gradio app\n"
	@printf "  ingest  Build/update the local vector index\n"
	@printf "  reindex Clear and rebuild the local vector index\n"
	@printf "  smoke-hash     Index samples with lightweight hash embeddings\n"
	@printf "  smoke-retrieve  Query the current index\n"
	@printf "  smoke-answer    Run deterministic end-to-end RAG smoke answers\n"
	@printf "  download-model  Download the configured GGUF from Hugging Face\n"
	@printf "  check-llama-cpp Check llama-cpp-python and model path\n"
	@printf "  docs    Build Sphinx documentation\n"
	@printf "  clean   Remove build and cache artifacts\n"

sync:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest

run:
	uv run python app.py

ingest:
	uv run python -m rag_gemma_local.ingest.pipeline

reindex:
	uv run python -m rag_gemma_local.ingest.reindex

smoke-hash:
	EMBEDDING_PROVIDER=hash CHROMA_DB_PATH=./data/indexes/smoke_hash uv run python -m rag_gemma_local.ingest.pipeline

smoke-retrieve:
	uv run python scripts/smoke_retrieve.py

smoke-answer:
	LLM_PROVIDER=context uv run python scripts/smoke_answer.py

download-model:
	uv run python scripts/download_model.py

check-llama-cpp:
	uv run python scripts/check_llama_cpp.py

docs:
	LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 uv run --group docs sphinx-build -b html docs/source docs/built

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
