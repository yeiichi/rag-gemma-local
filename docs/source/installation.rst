Installation
============

Requirements
------------

- macOS on Intel x86_64 for the current reference environment.
- Python 3.11 or 3.12.
- ``uv`` for dependency management.
- Enough disk space for a GGUF model file and local Chroma indexes.

Install dependencies
--------------------

From the project root:

.. code-block:: bash

   uv sync

To enable the local Gemma runtime:

.. code-block:: bash

   uv sync --extra llama-cpp

The reference machine builds ``llama-cpp-python`` locally. This can take several
minutes.

Download the model
------------------

Download the configured quantized Gemma GGUF:

.. code-block:: bash

   make download-model
   make check-llama-cpp

The model is stored under ``models/llm/`` and is intentionally ignored by Git.
