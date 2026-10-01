from itertools import pairwise

import pytest

from docassist.ingest.chunker import chunk_pages


def sentence(i: int, words: int = 10) -> str:
    return " ".join([f"s{i}w{j}" for j in range(words - 1)] + [f"end{i}."])


def test_short_pages_become_one_chunk_each_with_page_numbers():
    chunks = chunk_pages(["Page one text.", "", "Page three text."], 100, 10)
    assert [(c.ordinal, c.page, c.text) for c in chunks] == [
        (0, 1, "Page one text."),
        (1, 3, "Page three text."),
    ]


def test_chunks_never_cross_page_boundaries():
    pages = [" ".join(sentence(i) for i in range(5)), " ".join(sentence(i) for i in range(5, 8))]
    chunks = chunk_pages(pages, max_words=1000, overlap_words=10)
    assert [c.page for c in chunks] == [1, 2]
    assert "end4." in chunks[0].text and "end5." not in chunks[0].text
    assert chunks[1].text.startswith("s5w0")


def test_chunks_respect_max_words_and_overlap_whole_sentences():
    page = " ".join(sentence(i) for i in range(10))  # 10 sentences x 10 words
    chunks = chunk_pages([page], max_words=35, overlap_words=10)

    assert all(len(c.text.split()) <= 35 for c in chunks)
    # Each chunk after the first starts by repeating the previous chunk's last sentence.
    for prev, cur in pairwise(chunks):
        last = int(prev.text.split()[-1].removeprefix("end").rstrip("."))
        assert cur.text.startswith(sentence(last))
    # Every sentence ends up in some chunk.
    joined = " ".join(c.text for c in chunks)
    assert all(f"end{i}." in joined for i in range(10))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_overlap_is_exactly_one_sentence_when_it_fits():
    page = " ".join(sentence(i) for i in range(4))
    first, second = chunk_pages([page], max_words=30, overlap_words=10)[:2]
    assert first.text == " ".join(sentence(i) for i in range(3))
    assert second.text == " ".join(sentence(i) for i in (2, 3))


def test_sentence_longer_than_max_words_is_split_on_words():
    chunks = chunk_pages([sentence(0, words=25)], max_words=10, overlap_words=2)
    assert [len(c.text.split()) for c in chunks] == [10, 10, 5]


def test_wrapped_lines_are_joined_and_whitespace_normalized():
    page = "Please cancel at least\n24 hours   before your\nappointment. Next sentence."
    (chunk,) = chunk_pages([page], 100, 10)
    assert chunk.text == "Please cancel at least 24 hours before your appointment. Next sentence."


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages(["x."], max_words=10, overlap_words=10)
