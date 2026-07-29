Configuration
=============

Configuration is read from environment variables, with ``.env`` loaded during
local development.

Retrieval
---------

.. code-block:: bash

   CHROMA_DB_PATH=./data/indexes/chroma
   RAW_DOCS_PATH=./data/raw
   RAG_TOP_K=5
   RAG_FETCH_K=20
   RAG_MAX_CHUNKS_PER_SOURCE=2

``RAG_FETCH_K`` controls how many Chroma candidates are fetched before source
diversification. ``RAG_MAX_CHUNKS_PER_SOURCE`` limits how many chunks from one
source file can enter the final context.

Embeddings
----------

.. code-block:: bash

   EMBEDDING_PROVIDER=sentence_transformers
   EMBEDDING_MODEL=intfloat/multilingual-e5-small

Use ``EMBEDDING_PROVIDER=hash`` for dependency-light smoke tests.

LLM
---

.. code-block:: bash

   LLM_PROVIDER=llama_cpp
   GEMMA_GGUF_PATH=./models/llm/gemma-2b-it-q4.gguf
   LLAMA_CPP_N_CTX=4096
   LLAMA_CPP_N_THREADS=4
   LLAMA_CPP_MAX_TOKENS=512
   LLAMA_CPP_FORCE_CPU_DEVICES=true

Keep ``LLAMA_CPP_FORCE_CPU_DEVICES=true`` on the reference Intel iMac. It avoids
a Metal backend initialization failure by forcing an explicit CPU device list.
