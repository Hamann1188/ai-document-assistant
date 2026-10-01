FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.20 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/uploads \
    && chown app /app/uploads

# Dependencies first, so code changes don't invalidate this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

# Bake the embedding model into the image (before the code, so code changes don't
# re-download it). Must match EMBEDDING_MODEL in src/docassist/config.py; a test checks.
ARG EMBEDDING_MODEL=google/embeddinggemma-300m
COPY scripts/fetch_embedding_model.py /tmp/
# Hub downloads are owner-only (0600); the app runs as a non-root user.
RUN python /tmp/fetch_embedding_model.py "${EMBEDDING_MODEL}" /app/models/embedding \
    && chmod -R a+rX /app/models

COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN uv sync --locked --no-dev --no-editable

ENV DOCASSIST_EMBEDDING_MODEL_PATH=/app/models/embedding \
    DOCASSIST_EMBEDDING_WARMUP=true \
    DOCASSIST_UPLOAD_DIR=/app/uploads \
    HF_HUB_OFFLINE=1

USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn docassist.api.main:app --host 0.0.0.0 --port 8000"]
