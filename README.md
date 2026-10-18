# 🧪 TestFailIQ — AI Test Failure Investigation Assistant

A RAG-based engineering tool for investigating failed software tests. Paste a CI log, get an instant root-cause diagnosis with similar historical failures and actionable fix suggestions.

## Architecture

```
CI Log Input
    ↓
Log Parser (regex-based extraction)
    ↓
sentence-transformers (all-MiniLM-L6-v2, local)
    ↓
FAISS (vector search over historical failures)
    +
Knowledge Base (semantic search)
    ↓
RAG Prompt → Ollama (local LLM)
    ↓
Root Cause + Next Steps + Fix Snippet
    ↓
React Dashboard
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai) installed with a model pulled:
  ```
  ollama pull llama3.2
  ```

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The server starts at **http://localhost:8000**. On first run it auto-ingests the 15 synthetic CI logs.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at **http://localhost:5173**

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check + index size |
| POST | `/api/diagnose` | Full RAG pipeline diagnosis |
| POST | `/api/ingest` | Add a log to the vector store |
| GET | `/api/failures` | List all indexed failures |
| GET | `/api/failures/{id}` | Get a specific failure |
| GET | `/api/knowledge` | List knowledge base entries |
| POST | `/api/feedback` | Submit correct/incorrect feedback |
| GET | `/api/metrics` | Retrieval@k and resolution metrics |

## Changing the LLM

Edit `backend/rag/generator.py`:

```python
OLLAMA_MODEL = "llama3.2"   # Change to "mistral", "llama3", "codellama", etc.
```

## Evaluation Metrics

- **Retrieval@1** — % of queries where the top-1 result has the same error type
- **Retrieval@3** — % of queries where a correct result appears in top-3
- **Resolution Rate** — % of user-submitted diagnoses marked as "correct"

View live metrics at the **Evaluation** tab in the dashboard.
