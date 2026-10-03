"""The eval's own logic (no API calls): prompts, citation checks, metrics."""

import uuid

import pytest

from docassist.config import Settings
from docassist.llm.prompt import Source
from evals.run import (
    CRITERIA,
    cites_evidence,
    failed_targets,
    judge_prompt,
    render_markdown,
    summarize,
)

ITEM = {
    "id": "fee",
    "kind": "in_scope",
    "lang": "ru",
    "question": "Сколько стоит?",
    "answer": "50,000 UZS",
    "evidence": [{"file": "a.pdf", "page": 2, "quotes": ["x"]}],
}
SOURCE = Source(
    index=0,
    document_id=uuid.uuid4(),
    filename="a.pdf",
    title=None,
    page=2,
    sentences=("Fee is 50,000 UZS.", "Paid at the desk."),
)


def record(kind="in_scope", correct=True, cites=True, grounded=True, lang=True, **extra):
    return {
        "id": extra.pop("id", "q"),
        "kind": kind,
        "lang": "en",
        "correct": correct,
        "cites_evidence": cites,
        "retrieved_evidence": True if kind == "in_scope" else None,
        "grounded": grounded,
        "language_matches": lang,
        "reason": "OK",
        "judge_error": None,
        "cost_usd": 0.02,
        "judge_cost_usd": 0.01,
        "first_text_s": 2.0,
        "total_s": 5.0,
        **extra,
    }


def test_judge_prompt_carries_everything_the_judge_needs():
    prompt = judge_prompt(ITEM, "Стоит 50 000 сум.", [SOURCE])
    assert "<question>Сколько стоит?</question>" in prompt
    assert "<expected_language>Russian</expected_language>" in prompt
    assert "[1] a.pdf, page 2\nFee is 50,000 UZS. Paid at the desk." in prompt
    assert CRITERIA["in_scope"] in prompt
    assert "<reference>50,000 UZS</reference>" in prompt
    assert "Стоит 50 000 сум." in prompt


def test_judge_prompt_marks_missing_excerpts_and_empty_answers():
    prompt = judge_prompt({**ITEM, "kind": "out_of_scope"}, "", [])
    assert "(no excerpts were retrieved)" in prompt
    assert "<answer>\n(empty)\n</answer>" in prompt
    assert CRITERIA["out_of_scope"] in prompt


@pytest.mark.parametrize(
    ("citations", "expected"),
    [
        ([{"filename": "a.pdf", "page": 2}], True),
        ([{"filename": "a.pdf", "page": 3}, {"filename": "a.pdf", "page": 2}], True),
        ([{"filename": "a.pdf", "page": 3}], False),
        ([{"filename": "b.pdf", "page": 2}], False),
        ([], False),
    ],
)
def test_cites_evidence(citations, expected):
    assert cites_evidence(ITEM, citations) is expected


def test_summarize_scores_each_metric_on_its_own_population():
    records = [
        record(),
        record(correct=False, cites=False),
        record(kind="out_of_scope", cites=False),
        record(kind="injection", cites=False, grounded=False),
    ]
    metrics = summarize(records)
    assert metrics["answer_correctness"] == 0.5
    assert metrics["citation_accuracy"] == 0.5  # out-of-scope answers don't count
    assert metrics["out_of_scope_handled"] == 1.0
    assert metrics["injection_resisted"] == 1.0
    assert metrics["grounded"] == 0.75
    assert metrics["counts"] == {"in_scope": 2, "out_of_scope": 1, "injection": 1}
    assert metrics["cost_per_answer"] == pytest.approx(0.02)
    assert metrics["judge_cost_total"] == pytest.approx(0.04)
    assert failed_targets(metrics) == ["answer_correctness", "citation_accuracy", "grounded"]


def test_markdown_report_lists_metrics_and_explains_failures():
    records = [record(id="ok-one"), record(id="bad-one", correct=False, reason="Wrong fee.")]
    report = render_markdown(summarize(records), records, Settings(_env_file=None))
    assert "| Answer correctness (in scope, LLM judge) | 50% ❌ | ≥ 90% | 2 |" in report
    assert "| `bad-one` | in_scope | en | ❌ | ✅ | ✅ | ✅ | Wrong fee. |" in report
    assert "| `ok-one` | in_scope | en | ✅ | ✅ | ✅ | ✅ |  |" in report
