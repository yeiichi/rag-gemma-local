from __future__ import annotations

from rag_gemma_local.config import get_settings


def main() -> None:
    settings = get_settings()
    import llama_cpp

    print(f"llama-cpp-python: {llama_cpp.__version__}")
    if settings.gemma_gguf_path.is_file():
        print(f"Gemma GGUF found: {settings.gemma_gguf_path}")
    else:
        print(f"Gemma GGUF missing: {settings.gemma_gguf_path}")


if __name__ == "__main__":
    main()
