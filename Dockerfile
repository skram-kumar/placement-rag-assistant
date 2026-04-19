# ── Base image ────────────────────────────────────────────────────────────────
# Slim Python 3.11 image — small footprint, production-ready
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential needed for some Python packages that compile C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory inside the container ────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (Docker layer caching) ──────────────────
# Copying requirements first means Docker won't re-install packages
# on every build unless requirements.txt actually changed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY rag_engine.py .
COPY api/ ./api/
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/

# ── Environment variables ─────────────────────────────────────────────────────
# GROQ_API_KEY is passed in via docker-compose.yml or -e flag
# Never hardcode secrets in Dockerfile
ENV PYTHONUNBUFFERED=1

# ── Expose the port FastAPI runs on ──────────────────────────────────────────
EXPOSE 8000

# ── Start the FastAPI server ──────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]