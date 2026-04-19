"""
FastAPI layer for Placement RAG Assistant.
Wraps rag_engine.py into a REST API.

Endpoints:
  GET  /             - Health check
  GET  /health       - Detailed health status
  POST /query        - Ask a placement question
  POST /rebuild-index - Re-ingest the Excel dataset into ChromaDB
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App state — RAG chain loaded once at startup
# ---------------------------------------------------------------------------
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG chain once when the server starts."""
    logger.info("Starting up — initialising RAG chain...")
    try:
        from rag_engine import initialize_rag
        rag_chain, retriever = initialize_rag()
        app_state["rag_chain"] = rag_chain
        app_state["retriever"] = retriever
        app_state["ready"] = True
        logger.info("RAG chain initialised successfully.")
    except Exception as e:
        logger.error(f"Failed to initialise RAG chain: {e}")
        app_state["ready"] = False
        app_state["startup_error"] = str(e)
    yield
    # Shutdown
    logger.info("Shutting down.")
    app_state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Placement RAG Assistant API",
    description="REST API for querying the placement dataset using RAG (LangChain + Groq + ChromaDB).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        example="Which companies offer internships with CTC above 20 LPA?",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to return the retrieved source documents.",
    )


class SourceDocument(BaseModel):
    company: str
    role: str
    ctc: str
    min_cgpa: str
    internship: str
    sector: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[list[SourceDocument]] = None
    latency_ms: float


class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded" | "error"
    rag_ready: bool
    message: str


class RebuildResponse(BaseModel):
    status: str
    message: str
    records_loaded: Optional[int] = None

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
def root():
    return {"message": "Placement RAG Assistant API is running. Visit /docs for the Swagger UI."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    if not app_state.get("ready"):
        error = app_state.get("startup_error", "Unknown error during startup.")
        return HealthResponse(status="error", rag_ready=False, message=error)
    return HealthResponse(status="ok", rag_ready=True, message="All systems operational.")


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(payload: QueryRequest, request: Request):
    """
    Ask any placement-related question.
    The RAG chain retrieves relevant company records and answers using Groq LLM.
    """
    if not app_state.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="RAG chain is not ready. Check /health for details.",
        )

    logger.info(f"Query received: {payload.question!r}")
    start = time.perf_counter()

    try:
        result = app_state["rag_chain"].invoke(payload.question)
    except Exception as e:
        logger.error(f"RAG chain error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG chain error: {str(e)}")

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    answer = result["answer"]
    source_docs = result.get("source_docs", [])

    sources = None
    if payload.include_sources and source_docs:
        seen = set()
        sources = []
        for doc in source_docs:
            m = doc.metadata
            key = f"{m.get('company')}-{m.get('role')}"
            if key not in seen:
                seen.add(key)
                sources.append(
                    SourceDocument(
                        company=m.get("company", ""),
                        role=m.get("role", ""),
                        ctc=m.get("ctc", ""),
                        min_cgpa=m.get("cgpa", ""),
                        internship=m.get("internship", ""),
                        sector=m.get("sector", ""),
                    )
                )

    logger.info(f"Query answered in {elapsed_ms} ms. Sources returned: {len(sources) if sources else 0}")

    return QueryResponse(
        question=payload.question,
        answer=answer,
        sources=sources,
        latency_ms=elapsed_ms,
    )


@app.post("/rebuild-index", response_model=RebuildResponse, tags=["Admin"])
def rebuild_index():
    """
    Re-ingests the Excel dataset and rebuilds the ChromaDB vector store.
    Use this when the dataset is updated.
    """
    logger.info("Rebuild index triggered.")
    try:
        import shutil
        from rag_engine import (
            load_excel_as_documents,
            build_vectorstore,
            initialize_rag,
            CHROMA_DB_DIR,
            EXCEL_PATH,
        )

        # Remove existing ChromaDB
        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)
            logger.info("Existing ChromaDB removed.")

        documents = load_excel_as_documents(EXCEL_PATH)
        build_vectorstore(documents)

        # Reload chain into app state
        rag_chain, retriever = initialize_rag()
        app_state["rag_chain"] = rag_chain
        app_state["retriever"] = retriever
        app_state["ready"] = True

        logger.info(f"Index rebuilt with {len(documents)} records.")
        return RebuildResponse(
            status="success",
            message="Vector store rebuilt and RAG chain reloaded.",
            records_loaded=len(documents),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")