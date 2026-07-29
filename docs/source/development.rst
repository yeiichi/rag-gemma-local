Development
===========

Common commands
---------------

.. code-block:: bash

   make lint
   make test
   make docs

Documentation builds to:

.. code-block:: text

   docs/built/

Repository hygiene
------------------

The repository is intended to be safe for public GitHub hosting. Do not commit:

- ``.env``
- local Chroma indexes under ``data/indexes/``
- private source documents under ``data/raw/``
- model weights under ``models/llm/``

The sample Markdown files in ``data/raw/samples/`` are intentionally allowed.

Making the GitHub repository public
-----------------------------------

After reviewing the ignored files and committing only source/docs/sample files,
make the remote repository public from GitHub settings or with the GitHub CLI:

.. code-block:: bash

   gh repo edit --visibility public

Run ``git status --ignored`` first and confirm model weights and private data are
not staged.
