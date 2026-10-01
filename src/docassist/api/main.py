import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from docassist.api import documents, health, search
from docassist.config import EMBEDDING_MODEL, Settings, get_settings
from docassist.db.models import EMBEDDING_DIM
from docassist.db.session import create_engine, create_session_factory
from docassist.ingest.pipeline import fail_interrupted
from docassist.retrieval.embedder import Embedder, FastEmbedder

logger = logging.getLogger(__name__)
# Uvicorn configures only its own loggers; this makes the app's INFO logs visible.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")


def default_embedder(settings: Settings) -> Embedder:
    return FastEmbedder(
        EMBEDDING_MODEL,
        cache_dir=settings.embedding_cache_dir,
        model_path=settings.embedding_model_path,
    )


def create_app(
    settings: Settings | None = None,
    engine_factory: Callable[[Settings], AsyncEngine] = create_engine,
    embedder_factory: Callable[[Settings], Embedder] = default_embedder,
    recover_interrupted: bool = True,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = engine_factory(settings)
        app.state.session_factory = create_session_factory(app.state.engine)
        app.state.embedder = embedder_factory(settings)
        if app.state.embedder.dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"Embedder dimension {app.state.embedder.dim} != database column {EMBEDDING_DIM}"
            )
        if settings.embedding_warmup:
            await anyio.to_thread.run_sync(app.state.embedder.embed_query, "warm-up")
            logger.info("Embedding model loaded")
        # Single-instance deployment: anything still `processing` was cut off by a restart.
        if recover_interrupted:
            try:
                failed = await fail_interrupted(app.state.session_factory)
            except (OSError, SQLAlchemyError) as exc:
                logger.warning("Skipped interrupted-document recovery: %s", type(exc).__name__)
            else:
                if failed:
                    logger.warning("Marked %d interrupted documents as failed", failed)
        try:
            yield
        finally:
            await app.state.engine.dispose()

    app = FastAPI(title="AI Document Assistant", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    return app


app = create_app()
