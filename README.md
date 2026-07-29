# rag-gemma-local

[![Documentation Status](https://readthedocs.org/projects/rag-gemma-local/badge/?version=latest)](https://rag-gemma-local.readthedocs.io/en/latest/?badge=latest)

Tiny on-premise RAG scaffold for English and Japanese documents using local
embeddings, a local vector store, and a Gemma-compatible local LLM runtime.

This project mirrors the shape of `rag-experiment-01`, but moves OpenAI-specific
logic behind local provider adapters.

## First target stack

- LLM runtime: `llama.cpp` via `llama-cpp-python`, or an Ollama-compatible HTTP endpoint
- LLM model: quantized Gemma 2B GGUF
- Embeddings: multilingual Sentence Transformers model
- Vector store: Chroma persisted under `data/indexes/chroma`
- UI: Gradio

## Install

```bash
uv sync
```

Native packages such as `llama-cpp-python` may need local build tools. If this
Monterey Intel iMac struggles with the native build, use the Ollama-compatible
HTTP adapter against a separately managed local runtime.

## Configure

```bash
cp .env.example .env
```

Edit `.env` for your local model paths and runtime choice.

For quick pipeline checks without downloading an embedding model, use:

```bash
EMBEDDING_PROVIDER=hash CHROMA_DB_PATH=./data/indexes/smoke_hash make ingest
EMBEDDING_PROVIDER=hash CHROMA_DB_PATH=./data/indexes/smoke_hash make smoke-retrieve
```

For the real multilingual embedding path:

```bash
EMBEDDING_PROVIDER=sentence_transformers CHROMA_DB_PATH=./data/indexes/smoke_e5 make ingest
EMBEDDING_PROVIDER=sentence_transformers CHROMA_DB_PATH=./data/indexes/smoke_e5 make smoke-retrieve
```

To smoke-test the full RAG path before downloading Gemma:

```bash
LLM_PROVIDER=context EMBEDDING_PROVIDER=sentence_transformers CHROMA_DB_PATH=./data/indexes/smoke_e5 make smoke-answer
```

After the GGUF is present, smoke-test with Gemma:

```bash
LLM_PROVIDER=llama_cpp EMBEDDING_PROVIDER=sentence_transformers CHROMA_DB_PATH=./data/indexes/smoke_e5 LLAMA_CPP_N_CTX=2048 LLAMA_CPP_MAX_TOKENS=160 make smoke-answer
```

To enable the llama.cpp provider:

```bash
uv sync --extra llama-cpp
make check-llama-cpp
```

Place the Gemma GGUF file at the path configured by `GEMMA_GGUF_PATH`, for
example `models/llm/gemma-2b-it-q4.gguf`.

On Intel macOS, keep `LLAMA_CPP_FORCE_CPU_DEVICES=true`. This avoids a Metal
backend initialization failure on machines where Metal is present in the build
but not usable for inference.

To download the configured Hugging Face GGUF:

```bash
make download-model
make check-llama-cpp
```

If the repository requires license acceptance or authentication, log in with
the Hugging Face CLI first or download the file manually into `models/llm/`.

## Ingest documents

Place source files under:

```text
data/raw/
```

Then run:

```bash
make ingest
```

Retrieval fetches a wider candidate set and diversifies by source file before
building the prompt. Tune this with:

```bash
RAG_TOP_K=5
RAG_FETCH_K=20
RAG_MAX_CHUNKS_PER_SOURCE=2
```

When testing different source documents, rebuild the index from scratch:

```bash
make reindex
```

`reindex` removes only the configured `CHROMA_DB_PATH`, and refuses to remove
paths outside the project directory.

The repository includes a tiny bilingual sample corpus under
`data/raw/samples/`.

## Run

```bash
make run
```

## Development

```bash
make lint
make test
```
