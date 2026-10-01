"""Stream an answer from Claude with citations mapped back to document pages."""

import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import anthropic
from anthropic import AsyncAnthropic, omit

from docassist.config import Settings
from docassist.llm.prompt import SYSTEM_PROMPT, Source, build_user_content

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens: input, output, cache read (models this app is tuned for).
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
}

NOT_CONFIGURED = "Answering is not configured: set DOCASSIST_ANTHROPIC_API_KEY."
UPSTREAM_ERROR = "The AI service is unavailable right now. Please try again in a minute."
BAD_KEY = "The AI service rejected the API key. Check DOCASSIST_ANTHROPIC_API_KEY."


def source_payload(source: Source) -> dict:
    return {
        "index": source.index,
        "document_id": str(source.document_id),
        "filename": source.filename,
        "title": source.title,
        "page": source.page,
    }


def estimate_cost(model: str, usage) -> float | None:
    if model not in PRICES:
        return None
    input_price, output_price, cache_read_price = PRICES[model]
    cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0
    return (
        usage.input_tokens * input_price
        + cache_write * input_price * 1.25
        + cache_read * cache_read_price
        + usage.output_tokens * output_price
    ) / 1_000_000


async def stream_answer(
    client: AsyncAnthropic | None,
    settings: Settings,
    question: str,
    sources: Sequence[Source],
) -> AsyncIterator[dict]:
    """Yield answer events. Event `type`s:

    - sources: the pages sent to Claude (always first)
    - text: {block, text}: a piece of answer text block `block`
    - citation: {block, source, document_id, filename, page, cited_text}
    - reset: the requested model declined and a fallback model starts over;
      discard text and citations received so far
    - done: {stop_reason, model, usage, cost_usd} (stop_reason "refusal" or
      "max_tokens" means the answer is missing or cut short)
    - error: {message}: no answer; the message is safe to show
    """
    yield {"type": "sources", "sources": [source_payload(s) for s in sources]}
    if client is None:
        yield {"type": "error", "message": NOT_CONFIGURED}
        return

    use_fallback = settings.refusal_fallback
    try:
        async with client.beta.messages.stream(
            model=settings.model,
            max_tokens=settings.answer_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_content(question, sources)}],
            output_config={"effort": settings.answer_effort},
            betas=[FALLBACK_BETA] if use_fallback else omit,
            fallbacks="default" if use_fallback else omit,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_start" and event.content_block.type == "fallback":
                    yield {"type": "reset"}
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield {"type": "text", "block": event.index, "text": delta.text}
                    elif delta.type == "citations_delta":
                        citation = _map_citation(delta.citation, sources)
                        if citation is not None:
                            yield {"type": "citation", "block": event.index, **citation}
            message = await stream.get_final_message()
    except anthropic.AuthenticationError:
        logger.error("Claude API rejected the API key")
        yield {"type": "error", "message": BAD_KEY}
        return
    except anthropic.APIError as exc:
        logger.error("Claude API error: %s", type(exc).__name__, exc_info=True)
        yield {"type": "error", "message": UPSTREAM_ERROR}
        return

    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "cache_read_input_tokens": message.usage.cache_read_input_tokens or 0,
    }
    cost = estimate_cost(message.model, message.usage)
    logger.info(
        "Answered: model=%s stop=%s sources=%d in=%d out=%d cost=%s",
        message.model,
        message.stop_reason,
        len(sources),
        usage["input_tokens"],
        usage["output_tokens"],
        f"${cost:.4f}" if cost is not None else "n/a",
    )
    yield {
        "type": "done",
        "stop_reason": message.stop_reason,
        "model": message.model,
        "usage": usage,
        "cost_usd": cost,
    }


def _map_citation(citation, sources: Sequence[Source]) -> dict | None:
    if citation.type != "search_result_location":
        return None
    index = citation.search_result_index
    if not 0 <= index < len(sources):
        logger.warning("Citation points to unknown search result %s", index)
        return None
    source = sources[index]
    return {
        "source": index,
        "document_id": str(source.document_id),
        "filename": source.filename,
        "page": source.page,
        "cited_text": citation.cited_text,
    }


@dataclass
class AnswerBlock:
    text: str = ""
    citations: list[dict] = field(default_factory=list)


@dataclass
class Answer:
    sources: list[dict] = field(default_factory=list)
    blocks: dict[int, AnswerBlock] = field(default_factory=dict)
    stop_reason: str | None = None
    model: str | None = None
    usage: dict | None = None
    cost_usd: float | None = None
    error: str | None = None

    @property
    def text(self) -> str:
        return "".join(block.text for _, block in sorted(self.blocks.items()))

    @property
    def citations(self) -> list[dict]:
        return [c for _, block in sorted(self.blocks.items()) for c in block.citations]

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "blocks": [
                {"text": block.text, "citations": block.citations}
                for _, block in sorted(self.blocks.items())
            ],
            "sources": self.sources,
            "stop_reason": self.stop_reason,
            "model": self.model,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }


async def collect_answer(events: AsyncIterator[dict]) -> Answer:
    answer = Answer()
    async for event in events:
        match event["type"]:
            case "sources":
                answer.sources = event["sources"]
            case "text":
                answer.blocks.setdefault(event["block"], AnswerBlock()).text += event["text"]
            case "citation":
                citation = {k: v for k, v in event.items() if k not in ("type", "block")}
                answer.blocks.setdefault(event["block"], AnswerBlock()).citations.append(citation)
            case "reset":
                answer.blocks.clear()
            case "done":
                answer.stop_reason = event["stop_reason"]
                answer.model = event["model"]
                answer.usage = event["usage"]
                answer.cost_usd = event["cost_usd"]
            case "error":
                answer.error = event["message"]
    return answer
