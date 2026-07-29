from __future__ import annotations

import requests

from rag_gemma_local.config import Settings


class OllamaLLM:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"]
