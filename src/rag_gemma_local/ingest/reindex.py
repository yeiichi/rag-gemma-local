from __future__ import annotations

from rag_gemma_local.config import get_settings
from rag_gemma_local.ingest.pipeline import reindex


def main() -> None:
    reindex(get_settings())


if __name__ == "__main__":
    main()
