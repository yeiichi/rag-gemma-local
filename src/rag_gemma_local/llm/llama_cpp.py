from __future__ import annotations

import ctypes

from rag_gemma_local.config import Settings

_NULL_DEVICE_LIST = (ctypes.c_void_p * 1)(None)


def _force_cpu_devices(llama_cpp_module) -> None:
    original_default_params = llama_cpp_module.llama_model_default_params

    def cpu_model_params():
        params = original_default_params()
        params.devices = ctypes.cast(_NULL_DEVICE_LIST, ctypes.c_void_p)
        return params

    llama_cpp_module.llama_model_default_params = cpu_model_params


class LlamaCppLLM:
    def __init__(self, settings: Settings) -> None:
        if not settings.gemma_gguf_path.is_file():
            raise FileNotFoundError(
                "Gemma GGUF model not found at "
                f"{settings.gemma_gguf_path}. Set GEMMA_GGUF_PATH in .env."
            )
        try:
            import llama_cpp as llama_cpp_package
            import llama_cpp.llama as llama_module
            import llama_cpp.llama_cpp as llama_cpp_module
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install the optional extra "
                "or set LLM_PROVIDER=ollama."
            ) from exc

        if settings.llama_cpp_force_cpu_devices:
            _force_cpu_devices(llama_cpp_module)
            llama_module.llama_cpp.llama_model_default_params = (
                llama_cpp_module.llama_model_default_params
            )
            llama_cpp_package.llama_cpp.llama_model_default_params = (
                llama_cpp_module.llama_model_default_params
            )

        self._max_tokens = settings.llama_cpp_max_tokens
        self._repeat_penalty = settings.llama_cpp_repeat_penalty
        self._llm = Llama(
            model_path=str(settings.gemma_gguf_path),
            n_ctx=settings.llama_cpp_n_ctx,
            n_threads=settings.llama_cpp_n_threads,
            n_gpu_layers=0,
            offload_kqv=False,
            op_offload=False,
            verbose=False,
        )

    def generate(self, prompt: str) -> str:
        response = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
            temperature=0.0,
            repeat_penalty=self._repeat_penalty,
            stop=["Question:", "\n\nQuestion:"],
        )
        return response["choices"][0]["message"]["content"]
