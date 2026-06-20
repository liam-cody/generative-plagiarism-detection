import re
from functools import lru_cache

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=8192)
def split_sentences(text: str) -> tuple[str, ...]:
    """Split text into sentences, caching the result."""
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return ()
    return tuple(sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(normalized) if sentence.strip())
