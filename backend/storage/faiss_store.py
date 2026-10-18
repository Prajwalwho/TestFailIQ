"""
FAISS vector store for historical test failures.
Uses IndexFlatIP (inner product = cosine similarity on normalized vectors).
"""

import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Optional

import faiss
import numpy as np

from ingestion.embedder import embed_text

STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_store")
INDEX_PATH = os.path.join(STORE_DIR, "failures.index")
META_PATH = os.path.join(STORE_DIR, "failures_meta.pkl")

EMBEDDING_DIM = 384


@dataclass
class FailureRecord:
    id: str
    test_name: str
    module: str
    error_type: str
    error_message: str
    stacktrace: str
    log_text: str
    timestamp: str
    known_issue_id: Optional[str] = None
    resolution: Optional[str] = None
    feedback: Optional[str] = None  # "correct" | "incorrect" | None


class FAISSStore:
    def __init__(self):
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.records: list[FailureRecord] = []
        os.makedirs(STORE_DIR, exist_ok=True)

    def add(self, record: FailureRecord, search_text: str) -> None:
        """Embed search_text and add to FAISS index alongside the record."""
        vec = embed_text(search_text).reshape(1, -1).astype("float32")
        self.index.add(vec)
        self.records.append(record)

    def search(self, query_text: str, top_k: int = 5) -> list[tuple[FailureRecord, float]]:
        """Return top_k most similar failures with their similarity scores (0–1)."""
        if self.index.ntotal == 0:
            return []
        query_vec = embed_text(query_text).reshape(1, -1).astype("float32")
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.records[idx], float(score)))
        return results

    def get_all(self) -> list[FailureRecord]:
        return list(self.records)

    def get_by_id(self, failure_id: str) -> Optional[FailureRecord]:
        for r in self.records:
            if r.id == failure_id:
                return r
        return None

    def update_feedback(self, failure_id: str, feedback: str) -> bool:
        for r in self.records:
            if r.id == failure_id:
                r.feedback = feedback
                return True
        return False

    def save(self) -> None:
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.records, f)
        print(f"[FAISSStore] Saved {self.index.ntotal} vectors → {STORE_DIR}")

    def load(self) -> bool:
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.records = pickle.load(f)
            print(f"[FAISSStore] Loaded {self.index.ntotal} vectors from disk.")
            return True
        return False

    @property
    def size(self) -> int:
        return self.index.ntotal


# Singleton instance
_store: FAISSStore | None = None


def get_store() -> FAISSStore:
    global _store
    if _store is None:
        _store = FAISSStore()
        _store.load()
    return _store
