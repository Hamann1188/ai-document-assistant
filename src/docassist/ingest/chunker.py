import re
from dataclasses import dataclass

# Sentence end: terminal punctuation followed by whitespace. Newlines alone don't split,
# because pypdf breaks wrapped lines mid-sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    ordinal: int  # position within the document, from 0
    page: int  # 1-indexed
    text: str


def chunk_pages(pages: list[str], max_words: int, overlap_words: int) -> list[TextChunk]:
    """Split page texts into chunks that never cross a page boundary.

    Chunks are built from whole sentences up to `max_words`; the next chunk repeats
    trailing sentences of the previous one, up to `overlap_words`. A sentence longer
    than `max_words` is split on word boundaries.
    """
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    chunks: list[TextChunk] = []
    for page_number, page in enumerate(pages, start=1):
        for text in _chunk_page(page, max_words, overlap_words):
            chunks.append(TextChunk(ordinal=len(chunks), page=page_number, text=text))
    return chunks


def _chunk_page(text: str, max_words: int, overlap_words: int) -> list[str]:
    units = [piece for s in split_sentences(text) for piece in _split_long(s, max_words)]

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and _words([*current, unit]) > max_words:
            chunks.append(" ".join(current))
            current = _tail(current, overlap_words)
            # The overlap plus a long unit may still not fit; drop the overlap then.
            if _words([*current, unit]) > max_words:
                current = []
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_sentences(text: str) -> list[str]:
    """Sentences with whitespace collapsed; empty input gives no sentences."""
    return [" ".join(s.split()) for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def _split_long(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _tail(units: list[str], overlap_words: int) -> list[str]:
    tail: list[str] = []
    for unit in reversed(units):
        if _words([unit, *tail]) > overlap_words:
            break
        tail.insert(0, unit)
    return tail


def _words(units: list[str]) -> int:
    return sum(len(u.split()) for u in units)
