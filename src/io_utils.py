from tqdm.auto import tqdm
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar
from collections.abc import Iterable, Iterator


# JSON helpers
def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


# JSONL helpers
def iter_jsonl(path: Path) -> Iterator[Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield json.loads(text)


def load_jsonl(path: Path) -> list[Any]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path, items: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")


def append_jsonl(path: Path, item: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False))
        handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


T = TypeVar("T")


def iter_with_progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    leave: bool = False,
) -> Iterable[T]:
    return tqdm(iterable, desc=desc, total=total, leave=leave)
