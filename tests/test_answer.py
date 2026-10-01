import uuid

import anthropic
import httpx2
import pytest

from docassist.config import Settings
from docassist.llm.answer import (
    BAD_KEY,
    FALLBACK_BETA,
    NOT_CONFIGURED,
    UPSTREAM_ERROR,
    collect_answer,
    estimate_cost,
    stream_answer,
)
from docassist.llm.prompt import NO_RESULTS, SYSTEM_PROMPT, build_user_content, group_sources
from docassist.retrieval.search import SearchHit
from tests.conftest import ev_block_start, ev_cite, ev_text, ev_thinking, final_message

DOC_A, DOC_B = uuid.uuid4(), uuid.uuid4()


def hit(chunk_id: int, doc: uuid.UUID, page: int, text: str, name: str = "a.pdf") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=doc,
        filename=name,
        title=None,
        page=page,
        text=text,
        vector_rank=chunk_id,
        text_rank=None,
        exact_match=False,
    )


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


async def run(client, sources, question="Q?", **overrides):
    return await collect_answer(stream_answer(client, settings(**overrides), question, sources))


# --- prompt -----------------------------------------------------------------------


def test_group_sources_merges_chunks_by_page_in_rank_order_without_overlap():
    hits = [
        hit(11, DOC_A, 2, "Two. Three."),  # best-ranked page comes first
        hit(5, DOC_B, 1, "Other doc.", name="b.pdf"),
        hit(10, DOC_A, 2, "One. Two."),  # earlier chunk of the same page, overlapping
    ]
    sources = group_sources(hits)
    assert [(s.index, s.filename, s.page) for s in sources] == [(0, "a.pdf", 2), (1, "b.pdf", 1)]
    assert sources[0].sentences == ("One.", "Two.", "Three.")


def test_user_content_has_one_cited_search_result_per_page_then_the_question():
    sources = group_sources([hit(1, DOC_A, 3, "Price is 5. Valid now.")])
    blocks = build_user_content("How much?", sources)
    assert blocks == [
        {
            "type": "search_result",
            "source": f"doc:{DOC_A}#page=3",
            "title": "a.pdf, p. 3",
            "content": [
                {"type": "text", "text": "Price is 5."},
                {"type": "text", "text": "Valid now."},
            ],
            "citations": {"enabled": True},
        },
        {"type": "text", "text": "Question: How much?"},
    ]


def test_user_content_without_sources_says_so():
    assert build_user_content("Q?", []) == [
        {"type": "text", "text": NO_RESULTS},
        {"type": "text", "text": "Question: Q?"},
    ]


# --- request ---------------------------------------------------------------------


async def test_request_follows_opus_5_5_rules(fake_claude_cls):
    client = fake_claude_cls()
    await run(client, group_sources([hit(1, DOC_A, 1, "Fact.")]), question="What?")
    (request,) = client.requests
    assert request["model"] == "claude-opus-5-5"
    assert request["max_tokens"] == 8000
    assert request["output_config"] == {"effort": "medium"}  # no `format`: citations forbid it
    assert request["system"] == SYSTEM_PROMPT
    assert "thinking" not in request  # adaptive by default; disabling is a 400
    assert request["betas"] == [FALLBACK_BETA]
    assert request["fallbacks"] == "default"
    (message,) = request["messages"]
    assert message["role"] == "user"
    assert message["content"][-1] == {"type": "text", "text": "Question: What?"}


async def test_fallback_can_be_switched_off(fake_claude_cls):
    client = fake_claude_cls()
    await run(client, [], refusal_fallback=False)
    request = client.requests[0]
    assert request["betas"] is anthropic.omit and request["fallbacks"] is anthropic.omit


# --- events ------------------------------------------------------------------------


async def test_text_and_citations_are_mapped_to_pages(fake_claude_cls):
    sources = group_sources(
        [hit(1, DOC_A, 2, "Fee is 50,000 UZS."), hit(2, DOC_B, 4, "Other.", name="b.pdf")]
    )
    client = fake_claude_cls(
        events=[
            ev_block_start(0, "thinking"),
            ev_thinking(0),
            ev_block_start(1, "text"),
            ev_text(1, "The fee is "),
            ev_block_start(2, "text"),
            ev_text(2, "50,000 UZS"),
            ev_cite(2, 0, "Fee is 50,000 UZS."),
            ev_cite(2, 7, "out of range"),  # ignored, not a crash
            ev_text(3, "."),
        ],
        message=final_message(input_tokens=2000, output_tokens=300),
    )
    answer = await run(client, sources)

    assert answer.text == "The fee is 50,000 UZS."
    assert answer.citations == [
        {
            "source": 0,
            "document_id": str(DOC_A),
            "filename": "a.pdf",
            "page": 2,
            "cited_text": "Fee is 50,000 UZS.",
        }
    ]
    assert [s["page"] for s in answer.sources] == [2, 4]
    assert answer.stop_reason == "end_turn"
    assert answer.cost_usd == pytest.approx((2000 * 4 + 300 * 20) / 1e6)
    assert answer.error is None


async def test_fallback_block_resets_the_partial_answer(fake_claude_cls):
    client = fake_claude_cls(
        events=[
            ev_text(0, "Partial from the first model"),
            ev_block_start(1, "fallback"),
            ev_text(2, "Answer from the fallback model."),
        ],
        message=final_message(model="claude-opus-4-8"),
    )
    answer = await run(client, [])
    assert answer.text == "Answer from the fallback model."
    assert answer.model == "claude-opus-4-8"
    assert answer.cost_usd is None  # no price table for that model


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
async def test_incomplete_answers_report_the_stop_reason(fake_claude_cls, stop_reason):
    client = fake_claude_cls(message=final_message(stop_reason=stop_reason))
    assert (await run(client, [])).stop_reason == stop_reason


async def test_missing_api_key_gives_an_error_without_calling_claude():
    answer = await run(None, group_sources([hit(1, DOC_A, 1, "x.")]))
    assert answer.error == NOT_CONFIGURED
    assert len(answer.sources) == 1  # the sources are still reported


def _status_error(cls, status: int):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (_status_error(anthropic.AuthenticationError, 401), BAD_KEY),
        (_status_error(anthropic.InternalServerError, 500), UPSTREAM_ERROR),
        (
            anthropic.APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            ),
            UPSTREAM_ERROR,
        ),
    ],
)
async def test_api_errors_become_safe_messages(fake_claude_cls, error, message):
    answer = await run(fake_claude_cls(error=error), [])
    assert answer.error == message
    assert answer.text == ""


def test_estimate_cost_counts_cache_reads_and_writes():
    usage = final_message().usage
    usage.cache_read_input_tokens = 1_000_000
    usage.cache_creation_input_tokens = 1_000_000
    usage.input_tokens = usage.output_tokens = 0
    assert estimate_cost("claude-opus-5-5", usage) == pytest.approx(0.20 + 4 * 1.25)
    assert estimate_cost("some-other-model", usage) is None
