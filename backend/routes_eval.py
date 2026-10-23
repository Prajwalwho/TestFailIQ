"""
Read-only evaluation endpoints. They serve result files written by the evaluation scripts and never
rerun anything or touch the FAISS index.
"""

import json
import os

from fastapi import APIRouter, HTTPException

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EVAL_RESULTS_PATH = os.path.join(DATA_DIR, "eval_results.json")
REAL_FILES = {
    "retrieval": "eval_real_results.json",   # similar-failure retrieval on real logs
    "kb": "eval_real_kb_results.json",       # knowledge-base retrieval on real logs
    "rag": "eval_rag_results.json",          # RAG vs no-RAG ablation
    "thresholds": "thresholds.json",         # relevance thresholds and confidence calibration
}


def _read_optional(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@router.get("/api/eval-results")
def get_eval_results():
    """Held-out synthetic evaluation written by evaluation/run_eval.py."""
    if not os.path.exists(EVAL_RESULTS_PATH):
        raise HTTPException(status_code=404, detail="No eval results yet. Run: python evaluation/run_eval.py")
    try:
        with open(EVAL_RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"eval_results.json could not be read ({type(e).__name__})")


@router.get("/api/eval-real")
def get_eval_real():
    """Real-data results. A file that does not exist yet comes back as null."""
    return {key: _read_optional(os.path.join(DATA_DIR, name)) for key, name in REAL_FILES.items()}