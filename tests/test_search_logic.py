import pytest

from docassist.retrieval.search import build_tsquery, extract_identifiers


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What does clause 4.2 of the supply agreement say?", ["4.2"]),
        ("Show Supply Agreement No. 07/2026, item A-12 and N95 masks", ["07/2026", "A-12", "N95"]),
        ("Сколько стоит ОПТГ?", ["ОПТГ"]),
        ("Is VAT included? Is VAT extra?", ["VAT"]),
        # Plain numbers are values, not references; single capitals are words.
        ("What happens if I arrive 20 minutes late?", []),
        ("I need 50,000 UZS or 7 500 000", ["UZS"]),
        ("Klinikada bepul Wi-Fi bormi?", []),
        ("", []),
    ],
)
def test_extract_identifiers(query, expected):
    assert extract_identifiers(query) == expected


def test_build_tsquery_drops_stopwords_and_ors_the_rest():
    lexemes = ["what", "are", "your", "hours", "on", "saturday"]
    assert build_tsquery(lexemes) == "'hour':* | 'saturd':*"


def test_build_tsquery_prefixes_long_words_only():
    assert build_tsquery(["имплантации", "под", "ключ"]) == "'имплантац':* | 'ключ'"
    assert build_tsquery(["cancel"]) == "'canc':*"


def test_build_tsquery_keeps_numbers_and_codes_exact():
    assert build_tsquery(["4.2", "07/2026", "7", "оптг"]) == "'4.2' | '07/2026' | '7' | 'оптг'"


def test_build_tsquery_drops_single_letters_and_duplicates():
    assert build_tsquery(["o", "g", "il", "wi-fi", "wi-fi"]) == "'il' | 'wi-fi'"


def test_build_tsquery_escapes_quotes_and_backslashes():
    assert build_tsquery(["it's", "a\\b"]) == "'it''s' | 'a\\\\b'"


def test_build_tsquery_returns_none_when_nothing_is_left():
    assert build_tsquery(["what", "is", "the"]) is None
    assert build_tsquery([]) is None
