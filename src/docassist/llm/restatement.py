"""Spot cited quotes that only restate the sentence before them.

With citations on, the model sometimes states a fact in its own words and then
repeats it as a cited verbatim quote:

    You must cancel at least 24 hours ahead. Please cancel or reschedule at least
    24 hours before your appointment.[1]

The interface already shows the quoted source text in a footnote, so the second
copy is noise. The answer stream drops such a quote's text and keeps its citation,
which then attaches to the end of the previous sentence.
"""

import re
from collections.abc import Sequence

from docassist.retrieval.search import STOPWORDS

# "7 500 000", "50,000", "0.1", "24": one token each, separators removed.
_NUMBER = re.compile(r"\d{1,3}(?:[ , ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
_WORD = re.compile(r"\w+")
# A sentence ends at . ! ? followed by whitespace, or at a line break.
_BOUNDARY = re.compile(r"[.!?](?=\s)|\n")

MIN_TOKENS = 3
QUOTE_COVERAGE = 0.8  # share of the block's tokens found in the cited source text
RESTATEMENT_OVERLAP = 0.5  # shared tokens over the shorter of block and sentence


def content_tokens(text: str) -> set[str]:
    numbers = {re.sub(r"[ , ]", "", n) for n in _NUMBER.findall(text)}
    words = {
        w
        for w in _WORD.findall(_NUMBER.sub(" ", text).lower())
        if len(w) > 1 and w not in STOPWORDS
    }
    return words | numbers


def last_sentence_span(text: str) -> tuple[int, int] | None:
    """(start, end) of the sentence `text` ends with, or None mid-sentence."""
    end = len(text.rstrip())
    core = text[:end].rstrip("*_ ")  # closing Markdown emphasis after the period
    if not core or core[-1] not in ".!?":
        return None
    start = 0
    for match in _BOUNDARY.finditer(text, 0, len(core) - 1):
        start = match.end()
    while text[start].isspace():
        start += 1
    return start, end


def is_restatement(block_text: str, cited_texts: Sequence[str], sentence: str) -> bool:
    """A verbatim quote of its source that repeats `sentence`, the one before it."""
    block = content_tokens(block_text)
    previous = content_tokens(sentence)
    if len(block) < MIN_TOKENS or len(previous) < MIN_TOKENS:
        return False
    source: set[str] = set()
    for cited in cited_texts:
        source |= content_tokens(cited)
    if len(block & source) / len(block) < QUOTE_COVERAGE:
        return False
    return len(block & previous) / min(len(block), len(previous)) >= RESTATEMENT_OVERLAP
