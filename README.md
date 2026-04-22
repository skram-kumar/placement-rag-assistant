# Placement RAG Assistant

A conversational AI assistant that answers placement-related queries using Retrieval Augmented Generation (RAG). Built as a personal project to explore how LLMs can be grounded on structured private data — in this case, a placement dataset of 500 companies.

The project was extended with a production layer — REST API, containerization, and a CI/CD pipeline.

---

## What it does

Instead of the LLM guessing answers, every response is backed by actual retrieval from a vector database built on the placement dataset. Ask it anything — CTC ranges, CGPA cutoffs, which companies offer internships, sector-wise comparisons — and it pulls the relevant records before answering.

The "Sources" panel under each response shows exactly which company records were retrieved, so you can verify the answer isn't hallucinated. If the model doesn't have related content in the DB, it says so — it never makes up data.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq API — llama-3.3-70b-versatile |
| Embeddings | ChromaDB default (all-MiniLM-L6-v2 ONNX) |
| Vector DB | ChromaDB |
| RAG Framework | LangChain (LCEL) |
| UI | Streamlit |
| API | FastAPI |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Data | Excel — 500 companies, 10 sectors |

---

## Dataset

500 company records across 10 sectors — IT/Software, IT/Services, Product, Fintech, BFSI, Consulting, Analytics, Semiconductor, E-commerce, and Edtech. Each record has:

- Company Name, Role, CTC (LPA), No. of Openings, Internship (Yes/No), Min CGPA, Sector

---

## Architecture

```
Excel Dataset → ChromaDB (Vector Store) → LangChain RAG Chain → Groq LLM
                                                    ↓
                                             FastAPI Backend
                                            /              \
                                    Streamlit UI      REST API (/docs)
```

**MLOps flow:**
```
Push to GitHub → GitHub Actions (lint + test) → Build Docker Image → Push to Docker Hub
```

---

## Setup — Streamlit UI (Quick Start)

1. Clone the repo
```bash
git clone https://github.com/skram-kumar/placement-rag-assistant.git
cd placement-rag-assistant
```

2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add your Groq API key — create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Free API key at [console.groq.com](https://console.groq.com)

5. Run
```bash
streamlit run app.py
```

The vector database builds automatically on first run. Subsequent runs load it from disk instantly.

---

## Setup — FastAPI Backend

Run the REST API with auto-generated Swagger UI at `/docs`:

```bash
python -m uvicorn api.main:app --reload
```

API endpoints:
- `GET /health` — health check
- `POST /query` — ask a placement question
- `POST /rebuild-index` — re-ingest the dataset

---

## Setup — Docker

Run the entire app in a container:

```bash
docker compose up --build
```

Access the API at `http://localhost:8000/docs`

---

## CI/CD Pipeline

Every push to `main` triggers a GitHub Actions workflow with two jobs:
1. **Lint & Test** — flake8 linting + pytest on all API endpoints
2. **Build & Push** — builds Docker image and pushes to Docker Hub (only if tests pass)

---

## Sample Questions

- Which companies offer internships with CTC above 20 LPA?
- What is the CGPA cutoff for Google?
- List all fintech companies and their roles
- Which sector has the highest average CTC?
- Show me consulting companies that accept 7.0 CGPA

---

## Project Structure

```
placement-rag-assistant/
├── api/
│   └── main.py                 # FastAPI REST API
├── tests/
│   └── test_api.py             # Pytest tests for API endpoints
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
├── app.py                      # Streamlit chat UI
├── rag_engine.py               # RAG pipeline — retrieval, prompt, LLM chain
├── test_setup.py               # Pre-run checks for embedding, API key, dataset
├── Dockerfile                  # Container recipe
├── docker-compose.yml          # Container configuration
├── requirements.txt            # Python dependencies
├── .env                        # Not committed — add your own
└── data/
    └── placement_dataset.xlsx
```