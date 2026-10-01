"""Search against a real database.

The fake embedder makes the semantic ranking arbitrary (but deterministic), which is
what the identifier tests need: an exact code must come first whatever the vector
model thinks.
"""

from pathlib import Path

import pytest

SAMPLE_DOCS = Path(__file__).resolve().parents[2] / "sample_docs"


@pytest.fixture
def corpus(client):
    for name in ("price-list-ru.pdf", "supplier-agreement.pdf", "patient-handbook.pdf"):
        data = (SAMPLE_DOCS / name).read_bytes()
        response = client.post("/documents", files={"file": (name, data, "application/pdf")})
        assert response.status_code == 202
    return client


def hits(client, q: str, **params) -> list[dict]:
    response = client.get("/search", params={"q": q, **params})
    assert response.status_code == 200, response.text
    return response.json()["hits"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Сколько стоит ОПТГ?", ("price-list-ru.pdf", 1)),
        ("What does clause 4.2 say?", ("supplier-agreement.pdf", 2)),
    ],
)
def test_hybrid_puts_exact_identifier_matches_first(corpus, query, expected):
    result = hits(corpus, query, k=8)
    assert len(result) == 8
    assert (result[0]["filename"], result[0]["page"]) == expected
    assert result[0]["exact_match"] is True
    # Everything after the exact matches is in semantic order.
    rest = [h["vector_rank"] for h in result if not h["exact_match"]]
    assert rest == sorted(rest)


def test_identifier_does_not_match_inside_longer_numbers(corpus):
    # The corpus has "4.2." (clause) but no "4.25" or "14.2".
    assert not any(h["exact_match"] for h in hits(corpus, "clause 14.2"))


def test_unselective_identifier_is_ignored(corpus):
    # "UZS" is on most patient-handbook and price-list pages: not a useful filter.
    result = hits(corpus, "Prices in UZS", k=8)
    assert not any(h["exact_match"] for h in result)
    assert [h["vector_rank"] for h in result] == list(range(1, 9))


def test_vector_mode_is_pure_semantic_ranking(corpus):
    result = hits(corpus, "Сколько стоит ОПТГ?", k=5, mode="vector")
    assert [h["vector_rank"] for h in result] == [1, 2, 3, 4, 5]
    assert not any(h["exact_match"] or h["text_rank"] for h in result)


def test_text_mode_matches_word_forms_by_prefix(corpus):
    result = hits(corpus, "имплантации", mode="text")
    assert [(h["filename"], h["page"], h["text_rank"]) for h in result] == [
        ("price-list-ru.pdf", 3, 1)
    ]


def test_text_mode_with_only_stopwords_finds_nothing(corpus):
    assert hits(corpus, "what is the", mode="text") == []


def test_search_with_no_documents_returns_nothing(client):
    assert hits(client, "anything") == []


@pytest.mark.parametrize("params", [{"q": "   "}, {"q": "x", "k": 0}, {"q": "x", "k": 21}, {}])
def test_search_validates_parameters(client, params):
    assert client.get("/search", params=params).status_code == 422
