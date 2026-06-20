from pathlib import Path
from typing import Protocol

from src.data import QueryRecord


class RetrievalBackendConfig(Protocol):
    @property
    def model_name(self) -> str: ...

    def to_metadata(self) -> dict[str, object]: ...

    def index_id(self, corpus_path: Path) -> str: ...

    def ensure_index(self, corpus_path: Path, index_dir: Path) -> tuple[int, bool]: ...

    def retrieve(
        self,
        span_queries: list[QueryRecord],
        index_dir: Path,
        run_depth: int,
    ) -> dict[str, dict[str, float]]: ...

    def release_resources(self) -> None: ...


def resolve_index_dir(
    corpus_path: Path,
    index_root: Path,
    backend: RetrievalBackendConfig,
) -> Path:
    return index_root / backend.model_name / backend.index_id(corpus_path)
