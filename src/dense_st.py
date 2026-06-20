from dataclasses import dataclass, field
from functools import lru_cache
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from src.data import QueryRecord, load_corpus
from src.dense_utils import (
    DOCUMENT_EMBEDDINGS_FILENAME,
    DOCUMENTS_FILENAME,
    INDEX_MANIFEST_FILENAME,
    build_chunk_records,
    has_complete_index,
    load_doc_index_metadata,
    persist_doc_index_metadata,
    prepare_dense_score_inputs,
    score_query_batch,
)
from src.document_chunking import ChunkingConfig
from src.io_utils import file_sha256, read_json, write_json, write_jsonl
from src.io_utils import iter_with_progress


ENCODE_BATCH_SIZE = 16
SCORE_BATCH_SIZE = 64
DOCUMENT_REPRESENTATION = "full_document_chunks"


@dataclass(slots=True, frozen=True)
class _DenseStConfig:
    model_name: str
    trust_remote_code: bool = False
    encode_batch_size: int = ENCODE_BATCH_SIZE
    score_batch_size: int = SCORE_BATCH_SIZE
    score_device: str = "auto"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    def to_metadata(self) -> dict[str, object]:
        return {
            "document_representation": DOCUMENT_REPRESENTATION,
            "model_name": self.model_name,
            "chunking": self.chunking.to_metadata(),
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
        manifest_path = index_dir / INDEX_MANIFEST_FILENAME
        documents_path = index_dir / DOCUMENTS_FILENAME
        embeddings_path = index_dir / DOCUMENT_EMBEDDINGS_FILENAME

        if has_complete_index(index_dir):
            manifest = read_json(manifest_path)
            return int(manifest["corpus_count"]), False

        corpus = load_corpus(corpus_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        document_records = build_chunk_records(corpus, self.chunking)
        print(f"[{self.model_name}] Loaded {len(corpus)} documents")
        write_jsonl(documents_path, document_records)
        persist_doc_index_metadata(index_dir, document_records)
        print(f"[{self.model_name}] Persisted {len(document_records)} chunk records")

        encoder = _load_encoder(self.model_name, self.trust_remote_code)
        print(
            f"[{self.model_name}] Encoding {len(document_records)} chunk records with batch size {self.encode_batch_size}"
        )
        np.save(embeddings_path, _encode_document_records(encoder, document_records, self.encode_batch_size))
        print(f"[{self.model_name}] Finished encoding {len(document_records)} chunk records")

        write_json(
            manifest_path,
            {
                "index_id": index_dir.name,
                "corpus_count": len(corpus),
                "document_count": len(corpus),
                "chunk_count": len(document_records),
                **_index_metadata(corpus_path, self),
            },
            indent=2,
        )
        return len(corpus), True

    def retrieve(
        self,
        span_queries: list[QueryRecord],
        index_dir: Path,
        run_depth: int,
    ) -> dict[str, dict[str, float]]:
        if not span_queries:
            return {}

        document_ids, chunk_doc_indices = load_doc_index_metadata(index_dir)
        document_embeddings = np.load(
            index_dir / DOCUMENT_EMBEDDINGS_FILENAME,
            mmap_mode="r",
        )
        encoder = _load_encoder(self.model_name, self.trust_remote_code)
        query_embeddings = _encode_texts(encoder, [query.text for query in span_queries], self.encode_batch_size)
        score_device = _resolve_score_device(self.score_device)
        prepared_document_embeddings, prepared_chunk_doc_indices, active_score_device = _prepare_score_inputs(
            document_embeddings,
            chunk_doc_indices,
            score_device,
            self.model_name,
        )

        # Score queries in batches so the fixed document matrix is reused efficiently
        # while keeping the transient score matrix bounded on both CPU RAM and VRAM.
        batch_rankings_by_qid: dict[str, dict[str, float]] = {}
        batch_starts = range(0, len(span_queries), self.score_batch_size)
        total_batches = math.ceil(len(span_queries) / self.score_batch_size)
        for batch_start in iter_with_progress(
            batch_starts,
            desc=f"[{self.model_name}] Scoring dense span query batches on {active_score_device.type}",
            total=total_batches,
        ):
            batch_end = min(batch_start + self.score_batch_size, len(span_queries))
            batch_queries = span_queries[batch_start:batch_end]
            batch_rankings = score_query_batch(
                query_embeddings[batch_start:batch_end],
                prepared_document_embeddings,
                document_ids,
                prepared_chunk_doc_indices,
                run_depth,
            )
            batch_rankings_by_qid.update({query.qid: ranking for query, ranking in zip(batch_queries, batch_rankings)})

        return batch_rankings_by_qid

    def release_resources(self) -> None:
        _load_encoder.cache_clear()
        _clear_torch_memory()


def _index_metadata(corpus_path: Path, backend_config: _DenseStConfig) -> dict[str, object]:
    return {
        "backend": backend_config.to_metadata(),
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": file_sha256(corpus_path),
    }


def _encode_document_records(
    encoder: Any,
    document_records: list[dict[str, object]],
    batch_size: int,
) -> np.ndarray:
    if not document_records:
        return np.empty((0, 0), dtype=np.float32)
    return _encode_texts(encoder, [str(row["text"]) for row in document_records], batch_size)


@lru_cache(maxsize=2)
def _load_encoder(model_name: str, trust_remote_code: bool) -> Any:
    return SentenceTransformer(
        model_name,
        trust_remote_code=trust_remote_code,
        model_kwargs={"torch_dtype": torch.float16},
    )


def _encode_texts(
    encoder: Any,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings_array = np.asarray(embeddings, dtype=np.float32)
    if embeddings_array.ndim == 1:
        return embeddings_array.reshape(1, -1)
    return embeddings_array


def _clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _resolve_score_device(preferred_device: str) -> torch.device:
    if preferred_device == "cpu":
        return torch.device("cpu")
    if preferred_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[dense] CUDA scoring requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if preferred_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported dense score device: {preferred_device}")


def _prepare_score_inputs(
    document_embeddings: np.ndarray,
    chunk_doc_indices: np.ndarray,
    score_device: torch.device,
    model_name: str,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, torch.device]:
    # When CUDA is available, moving the fixed embedding matrix once is cheaper than
    # repeating a CPU-side full scan for every query batch. If the copy does not fit,
    # we fall back to CPU scoring instead of failing the experiment outright.
    if score_device.type != "cuda":
        prepared_document_embeddings, prepared_chunk_doc_indices = prepare_dense_score_inputs(
            document_embeddings,
            chunk_doc_indices,
            score_device=score_device,
        )
        return prepared_document_embeddings, prepared_chunk_doc_indices, score_device

    try:
        prepared_document_embeddings, prepared_chunk_doc_indices = prepare_dense_score_inputs(
            document_embeddings,
            chunk_doc_indices,
            score_device=score_device,
        )
        return prepared_document_embeddings, prepared_chunk_doc_indices, score_device
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        print(f"[{model_name}] Dense scoring did not fit in VRAM; falling back to CPU batching")
        _clear_torch_memory()
        cpu_device = torch.device("cpu")
        prepared_document_embeddings, prepared_chunk_doc_indices = prepare_dense_score_inputs(
            document_embeddings,
            chunk_doc_indices,
            score_device=cpu_device,
        )
        return prepared_document_embeddings, prepared_chunk_doc_indices, cpu_device


@dataclass(slots=True, frozen=True)
class BgeM3Config(_DenseStConfig):
    model_name: str = "BAAI/bge-m3"
    trust_remote_code: bool = False


@dataclass(slots=True, frozen=True)
class GteBaseConfig(_DenseStConfig):
    model_name: str = "Alibaba-NLP/gte-base-en-v1.5"
    trust_remote_code: bool = True


@dataclass(slots=True, frozen=True)
class Qwen3Embedding06BConfig(_DenseStConfig):
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    trust_remote_code: bool = True
    encode_batch_size: int = 16
