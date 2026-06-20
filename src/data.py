# Read corpus rows, queries, and qrels from the project data files

from dataclasses import dataclass, field
from pathlib import Path

from ranx import Qrels

from src.io_utils import iter_jsonl


@dataclass(slots=True)
class CorpusDocument:
    doc_id: str
    default_text: str


@dataclass(slots=True)
class QueryRecord:
    qid: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


def load_corpus(path: Path) -> list[CorpusDocument]:
    return [
        CorpusDocument(
            doc_id=str(row["doc_id"]),
            default_text=str(row["default_text"]),
        )
        for row in iter_jsonl(path)
    ]


def load_corpus_by_id(path: Path) -> dict[str, CorpusDocument]:
    return {document.doc_id: document for document in load_corpus(path)}


def load_queries(path: Path) -> list[QueryRecord]:
    return [
        QueryRecord(
            qid=str(row["qid"]),
            text=str(row["query"]),
            metadata={key: value for key, value in row.items() if key not in {"qid", "query"}},
        )
        for row in iter_jsonl(path)
    ]


def load_qrels(path: Path) -> Qrels:
    return Qrels.from_file(str(path))
