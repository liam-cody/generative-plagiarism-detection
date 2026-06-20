from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from ranx import Run
from ranx.fusion import rrf

from src.data import QueryRecord
from src.io_utils import read_json, write_json
from src.io_utils import iter_with_progress
from src.retrieval_backend import RetrievalBackendConfig, resolve_index_dir


MODEL_NAME = "rrf"
INDEX_MANIFEST_FILENAME = "manifest.json"


@dataclass(slots=True, frozen=True)
class RRFConfig:
    # Child retrieval backends to fuse
    children: tuple[RetrievalBackendConfig, ...]
    rrf_k: int = 60
    model_name: str = MODEL_NAME

    def __post_init__(self) -> None:
        # Normalize and validate child backends
        child_backends = tuple(self.children)
        if not child_backends:
            raise ValueError("RRFConfig requires at least one child backend")
        object.__setattr__(self, "children", child_backends)

    def to_metadata(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "rrf_k": self.rrf_k,
            "children": [child.to_metadata() for child in self.children],
        }

    def index_id(self, corpus_path: Path) -> str:
        fingerprint = json.dumps(
            _index_metadata(corpus_path, self),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]

    def ensure_index(
        self,
        corpus_path: Path,
        index_dir: Path,
    ) -> tuple[int, bool]:
        # Build/reuse child indices at their normal locations
        index_root = _resolve_index_root(index_dir)
        child_counts: list[int] = []
        created_any_child_index = False
        child_index_dirs: list[str] = []

        for child in self.children:
            child_index_dir = resolve_index_dir(corpus_path, index_root, child)
            corpus_count, created = child.ensure_index(corpus_path, child_index_dir)
            child_counts.append(corpus_count)
            created_any_child_index = created_any_child_index or created
            child_index_dirs.append(child_index_dir.relative_to(index_root).as_posix())

        index_dir.mkdir(parents=True, exist_ok=True)
        write_json(index_dir / INDEX_MANIFEST_FILENAME, {"children": child_index_dirs}, indent=2)
        return child_counts[0], created_any_child_index

    def retrieve(
        self,
        span_queries: list[QueryRecord],
        index_dir: Path,
        run_depth: int,
    ) -> dict[str, dict[str, float]]:
        # Load child index paths and collect child rankings
        manifest = read_json(index_dir / INDEX_MANIFEST_FILENAME)
        index_root = _resolve_index_root(index_dir)
        child_index_dirs = [index_root / Path(str(path)) for path in manifest["children"]]
        child_rankings = [
            child.retrieve(span_queries, child_index_dir, run_depth)
            for child, child_index_dir in zip(self.children, child_index_dirs)
        ]

        child_runs = [Run(rankings, name=child.model_name) for child, rankings in zip(self.children, child_rankings)]
        # Running RRF once across all span query ids keeps ranx responsible for the
        # fusion math while avoiding repeated tiny Run allocations and function calls.
        fused_run = rrf(
            child_runs,
            k=self.rrf_k,
        )
        fused_run_rankings = fused_run.to_dict()

        return {
            query.qid: _trim_results(fused_run_rankings.get(query.qid, {}), run_depth)
            for query in iter_with_progress(
                span_queries,
                desc="[rrf] Finalizing fused backend runs",
                total=len(span_queries),
            )
        }

    def release_resources(self) -> None:
        # Forward cleanup to child backends
        for child in self.children:
            child.release_resources()


def _index_metadata(corpus_path: Path, backend_config: RRFConfig) -> dict[str, object]:
    return {
        "backend": backend_config.to_metadata(),
        "children": [
            {
                "model_name": child.model_name,
                "index_id": child.index_id(corpus_path),
            }
            for child in backend_config.children
        ],
    }


def _resolve_index_root(index_dir: Path) -> Path:
    return index_dir.parents[1]


def _trim_results(results: dict[str, float], run_depth: int) -> dict[str, float]:
    ranked_results = sorted(results.items(), key=lambda item: (-item[1], item[0]))
    return {doc_id: score for doc_id, score in ranked_results[:run_depth]}
