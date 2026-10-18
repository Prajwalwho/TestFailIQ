"""
FastAPI application — Test Failure Investigation Assistant backend.
"""

import json
import os
import sys
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add backend dir to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from ingestion.log_parser import parse_log
from ingestion.embedder import embed_text
from storage.faiss_store import FailureRecord, get_store
from storage.knowledge_base import get_kb
from storage.diagnosis_store import get_diagnosis_store
from rag.retriever import retrieve
from rag.generator import generate_diagnosis
from evaluation.metrics import compute_metrics

app = FastAPI(
    title="Test Failure Investigation Assistant",
    description="RAG-based CI/test log analysis and root-cause diagnosis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Pre-load models and index synthetic logs on first run."""
    store = get_store()
    kb = get_kb()

    # Auto-ingest synthetic logs if store is empty
    if store.size == 0:
        _ingest_synthetic_logs(store)


def _ingest_synthetic_logs(store):
    """Ingest all synthetic logs from the metadata index."""
    meta_path = os.path.join(
        os.path.dirname(__file__), "data", "synthetic_logs_metadata.json"
    )
    if not os.path.exists(meta_path):
        print("[Startup] No synthetic logs found. Run data/generate_synthetic_logs.py first.")
        return

    with open(meta_path, encoding="utf-8") as f:
        logs = json.load(f)

    for log_data in logs:
        parsed = parse_log(
            log_data["log_text"],
            test_name=log_data.get("test_name"),
            module=log_data.get("module"),
        )
        record = FailureRecord(
            id=log_data["id"],
            test_name=parsed.test_name,
            module=parsed.module,
            error_type=parsed.error_type,
            error_message=parsed.error_message,
            stacktrace=parsed.stacktrace,
            log_text=log_data["log_text"],
            timestamp=log_data.get("timestamp", datetime.utcnow().isoformat()),
            known_issue_id=log_data.get("known_issue_id"),
            resolution=log_data.get("resolution"),
        )
        store.add(record, parsed.to_search_text())

    store.save()
    print(f"[Startup] Indexed {len(logs)} synthetic logs.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    log_text: str
    test_name: Optional[str] = None
    module: Optional[str] = None
    known_issue_id: Optional[str] = None
    resolution: Optional[str] = None


class DiagnoseRequest(BaseModel):
    log_text: str
    test_name: Optional[str] = None
    module: Optional[str] = None
    top_k: int = 5


class FeedbackRequest(BaseModel):
    diagnosis_id: str
    feedback: str  # "correct" | "incorrect"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    store = get_store()
    return {"status": "ok", "indexed_failures": store.size}


@app.post("/api/ingest")
def ingest(req: IngestRequest):
    """Ingest a new CI log into the vector store."""
    parsed = parse_log(req.log_text, test_name=req.test_name, module=req.module)
    record = FailureRecord(
        id=f"LOG-{uuid.uuid4().hex[:8].upper()}",
        test_name=parsed.test_name,
        module=parsed.module,
        error_type=parsed.error_type,
        error_message=parsed.error_message,
        stacktrace=parsed.stacktrace,
        log_text=req.log_text,
        timestamp=datetime.utcnow().isoformat(),
        known_issue_id=req.known_issue_id,
        resolution=req.resolution,
    )
    store = get_store()
    store.add(record, parsed.to_search_text())
    store.save()
    return {"success": True, "id": record.id, "parsed": {
        "test_name": parsed.test_name,
        "module": parsed.module,
        "error_type": parsed.error_type,
        "error_message": parsed.error_message,
    }}


@app.post("/api/diagnose")
def diagnose(req: DiagnoseRequest):
    """Run full RAG pipeline: parse → retrieve → generate diagnosis."""
    parsed = parse_log(req.log_text, test_name=req.test_name, module=req.module)
    evidence = retrieve(parsed, top_k=req.top_k)
    result = generate_diagnosis(parsed, evidence)

    top_match_id = evidence.similar_failures[0][0].id if evidence.similar_failures else None
    dx = get_diagnosis_store().add(
        test_name=parsed.test_name,
        module=parsed.module,
        error_type=parsed.error_type,
        root_cause=result.root_cause,
        confidence=result.confidence,
        model_used=result.model_used,
        top_match_id=top_match_id,
        kb_entry_used=result.kb_entry_used,
    )

    return {
        "diagnosis_id": dx.id,
        "parsed_log": {
            "test_name": parsed.test_name,
            "module": parsed.module,
            "error_type": parsed.error_type,
            "error_message": parsed.error_message,
        },
        "diagnosis": {
            "root_cause": result.root_cause,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "next_steps": result.next_steps,
            "fix_snippet": result.fix_snippet,
            "model_used": result.model_used,
        },
        "evidence": {
            "similar_failures": [
                {
                    "id": r.id,
                    "test_name": r.test_name,
                    "module": r.module,
                    "error_type": r.error_type,
                    "error_message": r.error_message,
                    "similarity": round(s, 3),
                    "timestamp": r.timestamp,
                    "resolution": r.resolution,
                }
                for r, s in evidence.similar_failures
            ],
            "kb_matches": [
                {
                    "id": e.id,
                    "title": e.title,
                    "error_type": e.error_type,
                    "root_cause": e.root_cause,
                    "resolution": e.resolution,
                    "fix_snippet": e.fix_snippet,
                    "similarity": round(s, 3),
                }
                for e, s in evidence.kb_matches
            ],
        },
    }


@app.get("/api/failures")
def list_failures():
    """Return all indexed failures."""
    store = get_store()
    return {
        "total": store.size,
        "failures": [
            {
                "id": r.id,
                "test_name": r.test_name,
                "module": r.module,
                "error_type": r.error_type,
                "error_message": r.error_message,
                "timestamp": r.timestamp,
                "resolution": r.resolution,
                "feedback": r.feedback,
            }
            for r in store.get_all()
        ],
    }


@app.get("/api/failures/{failure_id}")
def get_failure(failure_id: str):
    store = get_store()
    record = store.get_by_id(failure_id)
    if not record:
        raise HTTPException(status_code=404, detail="Failure not found")
    return {
        "id": record.id,
        "test_name": record.test_name,
        "module": record.module,
        "error_type": record.error_type,
        "error_message": record.error_message,
        "stacktrace": record.stacktrace,
        "log_text": record.log_text,
        "timestamp": record.timestamp,
        "known_issue_id": record.known_issue_id,
        "resolution": record.resolution,
        "feedback": record.feedback,
    }


@app.get("/api/knowledge")
def list_knowledge():
    """Return all knowledge base entries."""
    kb = get_kb()
    return {
        "total": len(kb.entries),
        "entries": [
            {
                "id": e.id,
                "title": e.title,
                "error_type": e.error_type,
                "module": e.module,
                "description": e.description,
                "root_cause": e.root_cause,
                "resolution": e.resolution,
                "fix_snippet": e.fix_snippet,
                "tags": e.tags,
            }
            for e in kb.get_all()
        ],
    }


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """Mark a diagnosis result as correct or incorrect."""
    if req.feedback not in ("correct", "incorrect"):
        raise HTTPException(status_code=400, detail="feedback must be 'correct' or 'incorrect'")
    updated = get_diagnosis_store().update_feedback(req.diagnosis_id, req.feedback)
    if not updated:
        raise HTTPException(status_code=404, detail="Diagnosis not found")
    return {"success": True, "diagnosis_id": req.diagnosis_id, "feedback": req.feedback}

    

@app.get("/api/metrics")
def get_metrics():
    """Return evaluation metrics."""
    metrics = compute_metrics()
    return {
        "total_indexed": metrics.total_indexed,
        "total_diagnoses": metrics.total_diagnoses,
        "feedback_correct": metrics.feedback_correct,
        "feedback_incorrect": metrics.feedback_incorrect,
        "feedback_pending": metrics.feedback_pending,
        "resolution_rate": metrics.resolution_rate,
        "retrieval_at_1": metrics.retrieval_at_1,
        "retrieval_at_3": metrics.retrieval_at_3,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
