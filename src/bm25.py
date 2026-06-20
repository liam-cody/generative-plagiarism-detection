from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import bm25s

from src.data import QueryRecord, load_corpus
from src.io_utils import file_sha256, read_json, write_json
from src.io_utils import iter_with_progress


INDEX_DIRNAME = "bm25s_index"
DOC_IDS_FILENAME = "doc_ids.json"
INDEX_MANIFEST_FILENAME = "manifest.json"
MODEL_NAME = "bm25"


@dataclass(slots=True, frozen=True)
class Bm25Config:
    method: str = "lucene"
    model_name: str = MODEL_NAME

    def to_metadata(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "method": self.method,
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
        index_path = index_dir / INDEX_DIRNAME
        doc_ids_path = index_dir / DOC_IDS_FILENAME
        manifest_path = index_dir / INDEX_MANIFEST_FILENAME

        if index_path.exists() and doc_ids_path.exists() and manifest_path.exists():
            manifest = read_json(manifest_path)
            return int(manifest["corpus_count"]), False

        corpus = load_corpus(corpus_path)
        doc_ids = [document.doc_id for document in corpus]
        texts = [document.default_text for document in corpus]
        index_metadata = _index_metadata(corpus_path, self)

        index_dir.mkdir(parents=True, exist_ok=True)

        retriever = bm25s.BM25(method=self.method)
        corpus_tokens = bm25s.tokenize(texts, show_progress=False, leave=False)
        retriever.index(corpus_tokens, show_progress=False, leave_progress=False)
        retriever.save(str(index_path))

        write_json(doc_ids_path, doc_ids, indent=2)
        write_json(
            manifest_path,
            {
                "index_id": index_dir.name,
                "corpus_count": len(doc_ids),
                **index_metadata,
            },
            indent=2,
        )
        return len(doc_ids), True

    def retrieve(
        self,
        span_queries: list[QueryRecord],
        index_dir: Path,
        run_depth: int,
    ) -> dict[str, dict[str, float]]:
        doc_ids_path = index_dir / DOC_IDS_FILENAME
        index_path = index_dir / INDEX_DIRNAME

        doc_ids = tuple(str(doc_id) for doc_id in read_json(doc_ids_path))
        retriever = bm25s.BM25.load(str(index_path), load_corpus=False)
        return {
            query.qid: _score_query(query.text, retriever, doc_ids, run_depth)
            for query in iter_with_progress(
                span_queries,
                desc=f"[{self.model_name}] Scoring span queries",
                total=len(span_queries),
            )
        }

    def release_resources(self) -> None:
        return


def _score_query(
    query_text: str,
    retriever: bm25s.BM25,
    doc_ids: tuple[str, ...],
    run_depth: int,
) -> dict[str, float]:
    query_tokens = bm25s.tokenize(query_text, show_progress=False, leave=False)
    docs, scores = retriever.retrieve(
        query_tokens,
        k=min(run_depth, len(doc_ids)),
        show_progress=False,
        leave_progress=False,
    )
    doc_indices = docs[0].tolist()
    doc_scores = scores[0].tolist()
    return {doc_ids[int(doc_index)]: float(score) for doc_index, score in zip(doc_indices, doc_scores)}


def _index_metadata(corpus_path: Path, backend_config: Bm25Config) -> dict[str, object]:
    return {
        "backend": backend_config.to_metadata(),
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": file_sha256(corpus_path),
    }
