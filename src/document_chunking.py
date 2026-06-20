import re
from dataclasses import dataclass
from src.text_utils import split_sentences

from src.data import CorpusDocument


DEFAULT_SECTION = "body"

_BLOCK_SPLIT_RE = re.compile(r"\n\s*\n+")
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+)\s+(.+)$", re.IGNORECASE)
_COMMON_SECTION_HEADINGS = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "background",
    "conclusion",
    "conclusions",
    "discussion",
    "experiments",
    "implementation",
    "introduction",
    "method",
    "methods",
    "related work",
    "results",
    "references",
}


@dataclass(slots=True, frozen=True)
class ChunkingConfig:
    # The corpus analysis showed paragraph-like blank-line blocks are usable,
    # but the long tail is large enough that we want a moderate target range plus
    # a smaller deterministic window fallback.
    min_words: int = 220
    max_words: int = 320
    window_size_words: int = 280
    window_overlap_words: int = 40

    def to_metadata(self) -> dict[str, int]:
        return {
            "min_words": self.min_words,
            "max_words": self.max_words,
            "window_size_words": self.window_size_words,
            "window_overlap_words": self.window_overlap_words,
        }


@dataclass(slots=True, frozen=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str
    chunk_index: int
    parent_block_index: int
    fallback_level: str


@dataclass(slots=True, frozen=True)
class _ChunkUnit:
    text: str
    section: str
    parent_block_index: int
    fallback_level: str


def build_document_chunks(
    document: CorpusDocument,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    # Start from blank-line paragraph blocks, carry forward heading context,
    # and merge small neighboring blocks within a section before we split long ones.
    chunking = ChunkingConfig() if config is None else config
    blocks = _split_blocks(document.default_text)
    section = DEFAULT_SECTION
    units: list[_ChunkUnit] = []
    pending_texts: list[str] = []
    pending_section = section
    pending_block_index = 0

    def flush_pending() -> None:
        nonlocal pending_texts
        if not pending_texts:
            return
        units.append(
            _ChunkUnit(
                text="\n\n".join(pending_texts).strip(),
                section=pending_section,
                parent_block_index=pending_block_index,
                fallback_level="none",
            )
        )
        pending_texts = []

    for block_index, block in enumerate(blocks):
        if is_heading_block(block):
            flush_pending()
            section = _normalize_heading(block)
            continue

        word_count = _word_count(block)
        if word_count > chunking.max_words:
            flush_pending()
            units.extend(_split_oversized_block(block, section, block_index, chunking))
            continue

        if not pending_texts:
            pending_texts = [block]
            pending_section = section
            pending_block_index = block_index
            continue

        pending_word_count = _word_count("\n\n".join(pending_texts))
        should_merge = pending_section == section and (
            pending_word_count < chunking.min_words or pending_word_count + word_count <= chunking.max_words
        )
        if should_merge:
            pending_texts.append(block)
            continue

        flush_pending()
        pending_texts = [block]
        pending_section = section
        pending_block_index = block_index

    flush_pending()

    return [
        DocumentChunk(
            chunk_id=f"{document.doc_id}::chunk-{chunk_index:04d}",
            doc_id=document.doc_id,
            text=unit.text,
            section=unit.section,
            chunk_index=chunk_index,
            parent_block_index=unit.parent_block_index,
            fallback_level=unit.fallback_level,
        )
        for chunk_index, unit in enumerate(units)
    ]


def is_heading_block(block: str) -> bool:
    # In this corpus, section markers usually survive as short standalone text blocks
    # rather than richer layout markup, so simple text heuristics work well here.
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) != 1:
        return False

    heading = _normalize_heading(lines[0])
    words = heading.split()
    if not 1 <= len(words) <= 12:
        return False
    if len(heading) > 80:
        return False
    if any(mark in heading for mark in ".!?"):
        return False
    if heading.lower() in _COMMON_SECTION_HEADINGS:
        return True

    capitalized_words = sum(word[:1].isupper() for word in words if word)
    return capitalized_words >= max(1, len(words) - 1)


def _split_blocks(text: str) -> list[str]:
    return [block.strip() for block in _BLOCK_SPLIT_RE.split(text.strip()) if block.strip()]


def _normalize_heading(block: str) -> str:
    heading = re.sub(r"\s+", " ", block.strip()).rstrip(":")
    numbered_match = _NUMBERED_HEADING_RE.match(heading)
    if numbered_match:
        heading = numbered_match.group(1).strip()
    if heading.isupper():
        return heading.title()
    return heading


def _split_oversized_block(
    text: str,
    section: str,
    parent_block_index: int,
    config: ChunkingConfig,
) -> list[_ChunkUnit]:
    # Oversized blocks are common enough in the corpus that we need a recursive fallback:
    # try sentence grouping first, then drop to overlapping word windows only if needed.
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return _split_with_windows(text, section, parent_block_index, config)

    units: list[_ChunkUnit] = []
    current_sentences: list[str] = []
    current_word_count = 0

    def flush_sentences() -> None:
        nonlocal current_sentences, current_word_count
        if not current_sentences:
            return
        units.append(
            _ChunkUnit(
                text=" ".join(current_sentences).strip(),
                section=section,
                parent_block_index=parent_block_index,
                fallback_level="sentence",
            )
        )
        current_sentences = []
        current_word_count = 0

    for sentence in sentences:
        sentence_word_count = _word_count(sentence)
        if sentence_word_count > config.max_words:
            flush_sentences()
            units.extend(_split_with_windows(sentence, section, parent_block_index, config))
            continue

        if current_sentences and current_word_count + sentence_word_count > config.max_words:
            flush_sentences()

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    flush_sentences()
    return units


def _split_with_windows(
    text: str,
    section: str,
    parent_block_index: int,
    config: ChunkingConfig,
) -> list[_ChunkUnit]:
    words = text.split()
    if not words:
        return []

    window_size = min(config.window_size_words, config.max_words)
    overlap = min(config.window_overlap_words, max(0, window_size - 1))
    step = max(1, window_size - overlap)
    units: list[_ChunkUnit] = []

    start = 0
    while start < len(words):
        units.append(
            _ChunkUnit(
                text=" ".join(words[start : start + window_size]).strip(),
                section=section,
                parent_block_index=parent_block_index,
                fallback_level="window",
            )
        )
        if start + window_size >= len(words):
            break
        start += step

    return units


def _word_count(text: str) -> int:
    return len(text.split())
