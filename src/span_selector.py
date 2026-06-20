from collections.abc import Callable
from functools import lru_cache
from src.text_utils import split_sentences


DEFAULT_SPAN_SIZE = 1


def select_sentence_spans(
    text: str,
    span_size: int = DEFAULT_SPAN_SIZE,
) -> list[str]:
    return list(_select_sentence_spans_cached(text, span_size))


@lru_cache(maxsize=8192)
def _select_sentence_spans_cached(
    text: str,
    span_size: int,
) -> tuple[str, ...]:
    if span_size <= 0:
        return ()

    sentences = split_sentences(text)
    if not sentences:
        return ()

    window_size = min(span_size, len(sentences))
    return tuple(
        " ".join(sentences[start : start + window_size]).strip() for start in range(len(sentences) - window_size + 1)
    )


def make_span_selector(
    span_size: int = DEFAULT_SPAN_SIZE,
) -> Callable[[str], list[str]]:
    """Build a sentence-window selector for ``ExperimentConfig.span_selector``."""

    def _select(text: str) -> list[str]:
        return select_sentence_spans(text, span_size=span_size)

    return _select
