import uuid

import pytest

from docassist.config import Settings
from docassist.llm.answer import collect_answer, stream_answer
from docassist.llm.prompt import group_sources
from docassist.llm.restatement import content_tokens, is_restatement, last_sentence_span
from docassist.retrieval.search import SearchHit
from tests.conftest import ev_block_start, ev_block_stop, ev_cite, ev_text

DOC = uuid.uuid4()
SOURCES = group_sources(
    [
        SearchHit(
            chunk_id=1,
            document_id=DOC,
            filename="patient-handbook.pdf",
            title=None,
            page=2,
            text="Please cancel or reschedule at least 24 hours before your appointment.",
            vector_rank=1,
            text_rank=None,
            exact_match=False,
        )
    ]
)
CANCEL = (
    "Cancellation policy Please cancel or reschedule at least 24 hours before your appointment."
)
FEE = "Late cancellations and missed appointments are charged a fee of 50,000 UZS."
PREPAY = "After two missed appointments, future bookings require a 100% prepayment."
CHANGES = "Цены могут быть изменены."
FINAL = "Окончательная стоимость указывается в плане лечения."


def cited_block(index: int, text: str, cited_text: str) -> list:
    return [
        ev_block_start(index, "text", cited=True),
        ev_cite(index, 0, cited_text),
        ev_text(index, text),
        ev_block_stop(index),
    ]


def plain_block(index: int, *pieces: str) -> list:
    return [
        ev_block_start(index, "text"),
        *(ev_text(index, piece) for piece in pieces),
        ev_block_stop(index),
    ]


async def answer_for(fake_claude_cls, events):
    client = fake_claude_cls(events=events)
    settings = Settings(_env_file=None)
    return await collect_answer(stream_answer(client, settings, "Q?", SOURCES))


# --- helpers ------------------------------------------------------------------------


def test_content_tokens_normalize_numbers_and_drop_stopwords():
    assert content_tokens("It costs 7 500 000 sum, not 50,000 UZS or 0.1%.") == {
        "costs", "7500000", "sum", "50000", "uzs", "0.1",
    }  # fmt: skip


@pytest.mark.parametrize(
    ("text", "sentence"),
    [
        ("First. You must cancel **24 hours** ahead. ", "You must cancel **24 hours** ahead."),
        ("You must cancel **24 hours ahead**. ", "You must cancel **24 hours ahead**."),
        ("Intro:\nFee applies.", "Fee applies."),
        ("One sentence only!", "One sentence only!"),
        ("Rate is 0.1% per day.", "Rate is 0.1% per day."),
    ],
)
def test_last_sentence_span_finds_the_finished_sentence(text, sentence):
    start, end = last_sentence_span(text)
    assert text[start:end] == sentence


@pytest.mark.parametrize("text", ["Also, ", "- **Fee:** ", "Учтите: ", ""])
def test_last_sentence_span_is_none_mid_sentence(text):
    assert last_sentence_span(text) is None


def test_is_restatement_needs_a_quote_that_repeats_the_sentence():
    paraphrase = "You need to cancel or reschedule at least 24 hours before your appointment."
    quote = "Please cancel or reschedule at least 24 hours before your appointment."
    assert is_restatement(quote, [CANCEL], paraphrase)
    # Same words, but the block isn't a quote of its source.
    assert not is_restatement(quote, ["Something else entirely, 99 dollars."], paraphrase)
    # A quote that adds a new fact.
    assert not is_restatement(FINAL, [FINAL], "Цены могут быть изменены.")
    # Different amounts are different facts.
    assert not is_restatement(
        "Установка импланта без коронки 5 000 000",
        ["Установка импланта без коронки 5 000 000"],
        "Имплантация под ключ стоит 7 500 000 сум.",
    )


# --- in the answer stream -------------------------------------------------------------


async def test_quotes_restating_the_previous_sentence_are_dropped(fake_claude_cls):
    # The event sequence recorded from the real API on 2026-10-03.
    events = [
        *plain_block(1, "You need to canc", "el or reschedule **", "at least 24 hours before",
                     " your appointment**. "),
        *cited_block(2, "Please cancel or reschedule at least 24 hours before your appointment.",
                     CANCEL),
        *plain_block(3, "\n\nIf you c", "ancel later than", " that, you'll be charged", " **",
                     "50,000 UZS**", ". "),
        *cited_block(4, FEE, FEE),
        *plain_block(5, " The same fee applies if you miss an appointment.\n\nAlso, "),
        *cited_block(6, "after two missed appointments, future bookings require a 100% "
                        "prepayment.", PREPAY),
    ]  # fmt: skip
    answer = await answer_for(fake_claude_cls, events)

    assert answer.text == (
        "You need to cancel or reschedule **at least 24 hours before your appointment**. "
        "\n\nIf you cancel later than that, you'll be charged **50,000 UZS**. "
        " The same fee applies if you miss an appointment.\n\n"
        "Also, after two missed appointments, future bookings require a 100% prepayment."
    )
    # Every citation survives; dropped quotes keep theirs on their (now empty) block.
    assert [c["cited_text"] for c in answer.citations] == [CANCEL, FEE, PREPAY]
    assert answer.blocks[2].text == "" and answer.blocks[2].citations


async def test_quotes_that_carry_the_answer_are_kept(fake_claude_cls):
    events = [
        *plain_block(0, "Учтите: "),
        *cited_block(1, CHANGES, CHANGES),
        *plain_block(2, " "),
        *cited_block(3, FINAL, FINAL),  # follows a cited sentence, so it's no paraphrase
        *plain_block(4, "\n\n- **Fee:** "),
        *cited_block(5, FEE, FEE),  # completes a list item
    ]
    answer = await answer_for(fake_claude_cls, events)
    assert answer.text == f"Учтите: {CHANGES} {FINAL}\n\n- **Fee:** {FEE}"
    assert len(answer.citations) == 3


async def test_uncited_text_streams_before_the_block_ends(fake_claude_cls):
    client = fake_claude_cls(
        events=[ev_block_start(0, "text"), ev_text(0, "Partial"), ev_block_stop(0)]
    )
    events = stream_answer(client, Settings(_env_file=None), "Q?", SOURCES)
    seen = [event["type"] async for event in events]
    assert seen == ["sources", "text", "done"]
