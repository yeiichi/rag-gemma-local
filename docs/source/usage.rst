Usage
=====

Add source documents
--------------------

Place Markdown, text, or PDF files under:

.. code-block:: text

   data/raw/

The repository includes sample English and Japanese Markdown files under
``data/raw/samples/``.

Rebuild the index
-----------------

Use ``reindex`` when testing different source documents:

.. code-block:: bash

   make reindex

For an explicit index path:

.. code-block:: bash

   EMBEDDING_PROVIDER=sentence_transformers \
   CHROMA_DB_PATH=./data/indexes/smoke_e5 \
   uv run python -m rag_gemma_local.ingest.reindex

Run the app
-----------

.. code-block:: bash

   LLM_PROVIDER=llama_cpp \
   EMBEDDING_PROVIDER=sentence_transformers \
   CHROMA_DB_PATH=./data/indexes/smoke_e5 \
   uv run python app.py

Open the Gradio URL printed by the command, usually ``http://localhost:7860``.

Smoke tests
-----------

Use hash embeddings for fast local checks:

.. code-block:: bash

   EMBEDDING_PROVIDER=hash CHROMA_DB_PATH=./data/indexes/smoke_hash make reindex
   EMBEDDING_PROVIDER=hash CHROMA_DB_PATH=./data/indexes/smoke_hash make smoke-retrieve

Use the deterministic context-only provider before testing Gemma:

.. code-block:: bash

   LLM_PROVIDER=context \
   EMBEDDING_PROVIDER=sentence_transformers \
   CHROMA_DB_PATH=./data/indexes/smoke_e5 \
   make smoke-answer
