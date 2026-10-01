from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from docassist.api import health
from docassist.config import Settings, get_settings
from docassist.db.session import create_engine


def create_app(
    settings: Settings | None = None,
    engine_factory: Callable[[Settings], AsyncEngine] = create_engine,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = engine_factory(settings)
        try:
            yield
        finally:
            await app.state.engine.dispose()

    app = FastAPI(title="AI Document Assistant", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(health.router)
    return app


app = create_app()
