from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from rag_gemma_local.config import get_settings


def main() -> None:
    settings = get_settings()
    settings.gemma_gguf_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=settings.hf_model_repo_id,
        filename=settings.hf_model_filename,
        local_dir=str(settings.gemma_gguf_path.parent),
        local_dir_use_symlinks=False,
    )
    if downloaded_path != str(settings.gemma_gguf_path):
        downloaded = Path(downloaded_path)
        if settings.gemma_gguf_path.exists():
            downloaded.unlink()
        else:
            downloaded.replace(settings.gemma_gguf_path)

    print(f"Downloaded {settings.hf_model_repo_id}/{settings.hf_model_filename}")
    print(f"Model path: {settings.gemma_gguf_path}")


if __name__ == "__main__":
    main()
