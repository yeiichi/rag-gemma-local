Architecture
============

Pipeline
--------

``rag-gemma-local`` follows a compact RAG pipeline:

.. code-block:: text

   local files
     -> loaders
     -> chunking
     -> embeddings
     -> Chroma index
     -> source-diverse retrieval
     -> prompt builder
     -> local LLM
     -> Gradio chat UI

Source-diverse retrieval
------------------------

The retriever fetches more candidates than the final context needs. It then
selects the best chunk from each source file before filling remaining slots.
This helps prevent Japanese prompts from targeting only Japanese documents, or
English prompts from targeting only English documents.

Prompting
---------

The prompt includes:

- a source inventory,
- retrieved context blocks,
- language guidance based on the user's question,
- a coverage requirement when multiple source files are present.

The answer should use retrieved facts only and cite source labels.

Local inference
---------------

The llama.cpp adapter loads a quantized Gemma 2B GGUF with CPU execution. On the
reference Intel macOS machine, the adapter forces an empty device list so the
runtime does not try to initialize an unusable Metal backend.
