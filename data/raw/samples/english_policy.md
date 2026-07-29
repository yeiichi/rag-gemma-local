# Local Development Environment Notes

The development environment uses uv with CPython 3.12.13 for macOS x86_64.
The project is pinned to Python 3.11 through 3.12 because the Intel iMac needs
older-compatible machine learning packages.

The embedding stack uses sentence-transformers 3.0.1, transformers 4.41.2,
Torch 2.2.2, and NumPy 1.26.4. NumPy is kept below version 2 to avoid binary
compatibility warnings from the installed Torch wheel.

The local language model is a quantized Gemma 2 2B instruction GGUF file stored
at `models/llm/gemma-2b-it-q4.gguf`. The llama-cpp-python package was built
locally, and the adapter forces an explicit CPU device list so the app avoids a
Metal initialization failure on this older Intel macOS machine.

The Gradio app runs locally at `localhost:7860`. Typical development commands
include `make reindex` to rebuild the Chroma index, `make test` for pytest, and
`make lint` for Ruff.
