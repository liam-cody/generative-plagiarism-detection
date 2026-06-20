"""Helpers for dense retrieval records, persistence, and score aggregation."""

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.data import CorpusDocument
from src.document_chunking import ChunkingConfig, build_document_chunks
from src.io_utils import iter_jsonl, read_json, write_json


DOCUMENT_EMBEDDINGS_FILENAME = "document_embeddings.npy"
DOCUMENTS_FILENAME = "documents.jsonl"
INDEX_MANIFEST_FILENAME = "manifest.json"
DOC_IDS_FILENAME = "doc_ids.json"
CHUNK_DOC_INDICES_FILENAME = "chunk_doc_indices.npy"


def has_complete_index(index_dir: Path) -> bool:
    required_paths = (
        index_dir / INDEX_MANIFEST_FILENAME,
        index_dir / DOCUMENTS_FILENAME,
        index_dir / DOCUMENT_EMBEDDINGS_FILENAME,
    )
    return all(path.exists() for path in required_paths)


def load_document_ids(documents_path: Path) -> tuple[str, ...]:
    return tuple(str(row["doc_id"]) for row in iter_jsonl(documents_path))


def persist_doc_index_metadata(
    index_dir: Path,
    document_records: list[dict[str, object]],
) -> None:
    # Dense retrieval only needs a compact chunk->document lookup at query time, so we
    # persist that metadata separately and avoid rereading the full chunk record file.
    doc_ids, chunk_doc_indices = _build_doc_index_metadata(document_records)
    write_json(index_dir / DOC_IDS_FILENAME, doc_ids, indent=2)
    np.save(index_dir / CHUNK_DOC_INDICES_FILENAME, chunk_doc_indices)


def load_doc_index_metadata(index_dir: Path) -> tuple[tuple[str, ...], np.ndarray]:
    doc_ids_path = index_dir / DOC_IDS_FILENAME
    chunk_doc_indices_path = index_dir / CHUNK_DOC_INDICES_FILENAME
    if not doc_ids_path.exists() or not chunk_doc_indices_path.exists():
        _backfill_doc_index_metadata(index_dir)

    doc_ids = tuple(str(doc_id) for doc_id in read_json(doc_ids_path))
    chunk_doc_indices = np.load(chunk_doc_indices_path, mmap_mode="r")
    return doc_ids, np.asarray(chunk_doc_indices, dtype=np.int64)


def build_chunk_records(
    corpus: Iterable[CorpusDocument],
    chunking: ChunkingConfig,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for document in corpus:
        records.extend(asdict(chunk) for chunk in build_document_chunks(document, chunking))
    return records


def score_query(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    document_ids: tuple[str, ...],
    run_depth: int,
) -> dict[str, float]:
    doc_ids, chunk_doc_indices = _build_doc_index_metadata({"doc_id": doc_id} for doc_id in document_ids)
    return score_query_batch(
        query_embedding.reshape(1, -1),
        document_embeddings,
        tuple(doc_ids),
        chunk_doc_indices,
        run_depth,
    )[0]


def prepare_dense_score_inputs(
    document_embeddings: np.ndarray,
    chunk_doc_indices: np.ndarray,
    *,
    score_device: torch.device,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    if score_device.type != "cuda":
        return np.asarray(document_embeddings), np.asarray(chunk_doc_indices, dtype=np.int64)

    return (
        torch.as_tensor(np.asarray(document_embeddings), device=score_device, dtype=torch.float32),
        torch.as_tensor(np.asarray(chunk_doc_indices), device=score_device, dtype=torch.int64),
    )


def score_query_batch(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray | torch.Tensor,
    doc_ids: tuple[str, ...],
    chunk_doc_indices: np.ndarray | torch.Tensor,
    run_depth: int,
) -> list[dict[str, float]]:
    if run_depth <= 0 or not doc_ids:
        return []

    query_embeddings_array = np.asarray(query_embeddings, dtype=np.float32)
    if query_embeddings_array.size == 0:
        return []
    if query_embeddings_array.ndim == 1:
        query_embeddings_array = query_embeddings_array.reshape(1, -1)

    if isinstance(document_embeddings, torch.Tensor):
        doc_scores = _aggregate_doc_scores_torch(
            query_embeddings_array,
            document_embeddings,
            chunk_doc_indices,
            len(doc_ids),
        )
    else:
        doc_scores = _aggregate_doc_scores_numpy(
            query_embeddings_array,
            np.asarray(document_embeddings),
            np.asarray(chunk_doc_indices, dtype=np.int64),
            len(doc_ids),
        )

    return [_rank_doc_scores(scores, doc_ids, run_depth) for scores in doc_scores]


def _backfill_doc_index_metadata(index_dir: Path) -> None:
    documents_path = index_dir / DOCUMENTS_FILENAME
    doc_ids, chunk_doc_indices = _build_doc_index_metadata(iter_jsonl(documents_path))
    write_json(index_dir / DOC_IDS_FILENAME, doc_ids, indent=2)
    np.save(index_dir / CHUNK_DOC_INDICES_FILENAME, chunk_doc_indices)


def _build_doc_index_metadata(document_rows: Iterable[object]) -> tuple[list[str], np.ndarray]:
    doc_index_by_id: dict[str, int] = {}
    doc_ids: list[str] = []
    chunk_doc_indices: list[int] = []

    for row in document_rows:
        doc_id = str(row["doc_id"])  # pyright: ignore[reportIndexIssue]
        doc_index = doc_index_by_id.get(doc_id)
        if doc_index is None:
            doc_index = len(doc_ids)
            doc_index_by_id[doc_id] = doc_index
            doc_ids.append(doc_id)
        chunk_doc_indices.append(doc_index)

    return doc_ids, np.asarray(chunk_doc_indices, dtype=np.int32)


def _aggregate_doc_scores_numpy(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    chunk_doc_indices: np.ndarray,
    doc_count: int,
) -> np.ndarray:
    chunk_scores = np.asarray(query_embeddings @ document_embeddings.T, dtype=np.float32)
    doc_scores = np.full((query_embeddings.shape[0], doc_count), -np.inf, dtype=np.float32)

    for row_index, query_chunk_scores in enumerate(chunk_scores):
        np.maximum.at(doc_scores[row_index], chunk_doc_indices, query_chunk_scores)

    return doc_scores


def _aggregate_doc_scores_torch(
    query_embeddings: np.ndarray,
    document_embeddings: torch.Tensor,
    chunk_doc_indices: np.ndarray | torch.Tensor,
    doc_count: int,
) -> np.ndarray:
    query_tensor = torch.as_tensor(query_embeddings, device=document_embeddings.device, dtype=document_embeddings.dtype)
    chunk_doc_indices_tensor = (
        chunk_doc_indices
        if isinstance(chunk_doc_indices, torch.Tensor)
        else torch.as_tensor(chunk_doc_indices, device=document_embeddings.device, dtype=torch.int64)
    )

    chunk_scores = query_tensor @ document_embeddings.T
    doc_scores = torch.full(
        (query_tensor.shape[0], doc_count),
        float("-inf"),
        device=document_embeddings.device,
        dtype=chunk_scores.dtype,
    )
    scatter_index = chunk_doc_indices_tensor.unsqueeze(0).expand(query_tensor.shape[0], -1)
    doc_scores.scatter_reduce_(1, scatter_index, chunk_scores, reduce="amax", include_self=True)
    return doc_scores.to(dtype=torch.float32).cpu().numpy()


def _rank_doc_scores(
    doc_scores: np.ndarray,
    doc_ids: tuple[str, ...],
    run_depth: int,
) -> dict[str, float]:
    top_count = min(run_depth, len(doc_ids))
    if top_count <= 0:
        return {}

    if top_count == len(doc_ids):
        candidate_indices = np.arange(len(doc_ids))
    else:
        top_indices = np.argpartition(doc_scores, -top_count)[-top_count:]
        # Keep every document tied at the cutoff so the final doc-id tiebreak matches
        # the previous full-sort behavior while still avoiding unnecessary global sorts.
        cutoff_score = float(doc_scores[top_indices].min())
        candidate_indices = np.flatnonzero(doc_scores >= cutoff_score)

    ranked_results = sorted(
        ((doc_ids[int(index)], float(doc_scores[int(index)])) for index in candidate_indices),
        key=lambda item: (-item[1], item[0]),
    )
    return {doc_id: score for doc_id, score in ranked_results[:top_count]}
