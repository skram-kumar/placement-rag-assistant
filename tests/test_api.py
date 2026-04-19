"""
Integration tests for the Placement RAG Assistant API.
Uses FastAPI's TestClient so no running server needed.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    """
    Patches rag_engine.initialize_rag so the app starts without
    needing a real Groq API key or ChromaDB during CI.
    """
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "Mock answer for testing.",
        "source_docs": [
            MagicMock(
                metadata={
                    "company": "TestCorp",
                    "role": "Engineer",
                    "ctc": "20",
                    "cgpa": "7.0",
                    "internship": "Yes",
                    "sector": "IT/Software",
                }
            )
        ],
    }
    mock_retriever = MagicMock()

    from api.main import app
    import api.main as main_module
    main_module.app_state.clear()
    main_module.app_state.update({
        "rag_chain": mock_chain,
        "retriever": mock_retriever,
        "ready": True
    })

    yield TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_root(client):
    """Root endpoint should return 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health(client):
    """Health endpoint should return ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["rag_ready"] is True
    assert data["status"] == "ok"


def test_query_valid(client):
    """Valid query should return answer and sources."""
    response = client.post(
        "/query",
        json={
            "question": "Which companies offer internships?",
            "include_sources": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "latency_ms" in data
    assert data["answer"] == "Mock answer for testing."


def test_query_too_short(client):
    """Question under 3 characters should return 422 validation error."""
    response = client.post(
        "/query",
        json={"question": "Hi", "include_sources": True},
    )
    assert response.status_code == 422


def test_query_no_sources(client):
    """include_sources=False should return null sources."""
    response = client.post(
        "/query",
        json={
            "question": "List all fintech companies",
            "include_sources": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["sources"] is None


def test_query_missing_field(client):
    """Missing question field should return 422."""
    response = client.post("/query", json={"include_sources": True})
    assert response.status_code == 422