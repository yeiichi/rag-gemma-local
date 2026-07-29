from __future__ import annotations

from rag_gemma_local.config import Settings
from rag_gemma_local.llm.base import LLM
from rag_gemma_local.llm.context import ContextOnlyLLM
from rag_gemma_local.llm.llama_cpp import LlamaCppLLM
from rag_gemma_local.llm.ollama import OllamaLLM


def get_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider.lower().strip()
    if provider == "context":
        return ContextOnlyLLM()
    if provider == "llama_cpp":
        return LlamaCppLLM(settings)
    if provider == "ollama":
        return OllamaLLM(settings)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
