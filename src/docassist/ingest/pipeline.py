"""Upload registration and background processing of PDF documents."""

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import anyio
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from docassist.config import Settings
from docassist.db.models import Chunk, Document, DocumentStatus
from docassist.ingest.chunker import chunk_pages
from docassist.ingest.pdf import PdfError, extract_pdf
from docassist.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)

INTERNAL_ERROR = "Internal error while processing the document. Upload it again to retry."


@dataclass(frozen=True)
class Registration:
    document: Document
    needs_processing: bool


def stored_path(settings: Settings, sha256: str) -> Path:
    return settings.upload_dir / f"{sha256}.pdf"


def clean_filename(name: str | None) -> str:
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return (base or "document.pdf")[:255]


async def register_upload(
    session: AsyncSession, settings: Settings, filename: str | None, data: bytes
) -> Registration:
    """Store the file and its document row. Identical files are stored once.

    Re-uploading a file whose processing failed retries it.
    """
    sha256 = hashlib.sha256(data).hexdigest()
    existing = await session.scalar(select(Document).where(Document.sha256 == sha256))
    if existing is not None:
        if existing.status != DocumentStatus.FAILED:
            return Registration(existing, needs_processing=False)
        existing.status, existing.error = DocumentStatus.PROCESSING, None
        await session.commit()
        return Registration(existing, needs_processing=True)

    path = stored_path(settings, sha256)
    await anyio.Path(path.parent).mkdir(parents=True, exist_ok=True)
    await anyio.Path(path).write_bytes(data)

    document = Document(filename=clean_filename(filename), sha256=sha256)
    session.add(document)
    try:
        await session.commit()
    except IntegrityError:
        # The same file was registered concurrently; use that row.
        await session.rollback()
        existing = await session.scalar(select(Document).where(Document.sha256 == sha256))
        return Registration(existing, needs_processing=False)
    return Registration(document, needs_processing=True)


async def process_document(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    settings: Settings,
    document_id: uuid.UUID,
) -> None:
    """Extract, chunk and embed a registered document, then mark it ready or failed."""
    async with session_factory() as session:
        document = await session.get(Document, document_id)
        if document is None:
            return
        try:
            data = await anyio.Path(stored_path(settings, document.sha256)).read_bytes()
            pdf = await anyio.to_thread.run_sync(extract_pdf, data, settings.max_pages)
            chunks = chunk_pages(pdf.pages, settings.chunk_max_words, settings.chunk_overlap_words)
            title = pdf.title or Path(document.filename).stem
            vectors = await anyio.to_thread.run_sync(
                embedder.embed_passages, [(title, chunk.text) for chunk in chunks]
            )
            session.add_all(
                Chunk(
                    document_id=document.id,
                    ordinal=chunk.ordinal,
                    page=chunk.page,
                    text=chunk.text,
                    embedding=vector,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            )
            document.title, document.page_count = pdf.title, len(pdf.pages)
            document.status = DocumentStatus.READY
            await session.commit()
            logger.info(
                "Ingested %s: %d pages, %d chunks", document.id, len(pdf.pages), len(chunks)
            )
        except Exception as exc:
            await session.rollback()
            if not isinstance(exc, PdfError):
                logger.exception("Failed to process document %s", document_id)
            await _mark_failed(
                session, document_id, str(exc) if isinstance(exc, PdfError) else None
            )


async def _mark_failed(session: AsyncSession, document_id: uuid.UUID, error: str | None) -> None:
    await session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(status=DocumentStatus.FAILED, error=error or INTERNAL_ERROR)
    )
    await session.commit()


async def fail_interrupted(session_factory: async_sessionmaker[AsyncSession]) -> int:
    """Mark documents left in `processing` by a previous run as failed (retry by re-upload)."""
    async with session_factory() as session:
        result = await session.execute(
            update(Document)
            .where(Document.status == DocumentStatus.PROCESSING)
            .values(status=DocumentStatus.FAILED, error=INTERNAL_ERROR)
        )
        await session.commit()
        return result.rowcount
