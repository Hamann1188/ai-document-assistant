"""Consistency checks between sample_docs/ and evals/questions.yaml.

If the generator content changes, these tests catch eval items whose evidence moved
or disappeared.
"""

import re
from collections import Counter
from functools import cache
from pathlib import Path

import pytest
import yaml
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "sample_docs"
QUESTIONS = yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text(encoding="utf-8"))

EXPECTED_PAGES = {
    "patient-handbook.pdf": 5,
    "price-list-ru.pdf": 3,
    "employee-handbook.pdf": 4,
    "supplier-agreement.pdf": 4,
}
KINDS = {"in_scope", "out_of_scope", "injection"}
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@cache
def page_texts(filename: str) -> tuple[str, ...]:
    return tuple(page.extract_text() for page in PdfReader(DOCS / filename).pages)


def normalize(text: str) -> str:
    return " ".join(text.split())


def test_corpus_files_and_page_counts():
    assert {p.name for p in DOCS.glob("*.pdf")} == set(EXPECTED_PAGES)
    for filename, pages in EXPECTED_PAGES.items():
        assert len(page_texts(filename)) == pages, filename


@pytest.mark.parametrize("filename", sorted(EXPECTED_PAGES))
def test_every_page_is_marked_fictional_and_extracts_cleanly(filename):
    for number, text in enumerate(page_texts(filename), start=1):
        assert "Fictional demo document" in text, f"{filename} p{number}"
        assert not CONTROL_CHARS.search(text), f"{filename} p{number} has control characters"


def test_question_ids_unique_and_fields_valid():
    ids = Counter(item["id"] for item in QUESTIONS)
    assert [i for i, n in ids.items() if n > 1] == []
    for item in QUESTIONS:
        assert item["kind"] in KINDS, item["id"]
        assert item["lang"] in {"en", "ru", "uz"}, item["id"]
        assert item["question"].strip() and item["answer"].strip(), item["id"]
        has_evidence = bool(item.get("evidence"))
        assert has_evidence == (item["kind"] != "out_of_scope"), item["id"]


def test_eval_set_covers_every_document_and_case_type():
    kinds = Counter(item["kind"] for item in QUESTIONS)
    assert kinds["in_scope"] >= 15
    assert kinds["out_of_scope"] >= 3
    assert kinds["injection"] >= 1
    cited = {ev["file"] for item in QUESTIONS for ev in item.get("evidence", [])}
    assert cited == set(EXPECTED_PAGES)


@pytest.mark.parametrize(
    ("item_id", "evidence"),
    [(item["id"], ev) for item in QUESTIONS for ev in item.get("evidence", [])],
)
def test_evidence_quotes_are_on_the_cited_page(item_id, evidence):
    pages = page_texts(evidence["file"])
    assert 1 <= evidence["page"] <= len(pages), item_id
    text = normalize(pages[evidence["page"] - 1])
    for quote in evidence["quotes"]:
        assert normalize(quote) in text, f"{item_id}: {quote!r} not on p{evidence['page']}"
