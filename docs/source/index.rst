.. _index:

rag-gemma-local documentation
=============================

Welcome to the documentation for ``rag-gemma-local``.

``rag-gemma-local`` is a tiny local RAG application for English and Japanese
documents. It runs a Gradio chat UI, indexes local files with multilingual
embeddings, stores vectors in Chroma, and answers with a quantized Gemma 2B GGUF
model through ``llama-cpp-python``.

Project goals
-------------

- Keep private documents local by default.
- Support English and Japanese source documents.
- Retrieve language-agnostic context across source files.
- Run on an older Intel iMac with macOS Monterey and 24 GB of memory.
- Provide a small, understandable codebase for experiments.
- Avoid committing local indexes, private documents, or model weights.

Current capabilities
--------------------

- Markdown, text, and PDF ingestion.
- Clean ``reindex`` command for testing different source documents.
- Source-diverse retrieval so one language or file does not dominate context.
- ``sentence-transformers`` embeddings for multilingual retrieval.
- Lightweight hash embeddings for smoke tests.
- Local Gemma inference through ``llama.cpp``.
- CPU-device workaround for Intel macOS systems where Metal initialization
  fails.
- Gradio UI at ``localhost:7860``.

Getting started
---------------

.. toctree::
   :maxdepth: 2

   installation
   configuration
   usage
   architecture
   api/index
   development
