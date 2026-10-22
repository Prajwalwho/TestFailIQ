"""
Knowledge base loader and search.
Loads known issues + fixes from knowledge_base.json and supports semantic search.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ingestion.embedder import embed_batch, embed_text

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")


@dataclass
class KnowledgeEntry:
    id: str
    title: str
    error_type: str
    module: str
    description: str
    root_cause: str
    resolution: str
    fix_snippet: str
    tags: list[str]
    category: str = ""  # coarse failure class, used by the real-data evaluation

    def to_search_text(self) -> str:
        return (
            f"{self.title}\n"
            f"Error: {self.error_type}\n"
            f"Module: {self.module}\n"
            f"Description: {self.description}\n"
            f"Root cause: {self.root_cause}\n"
            f"Tags: {', '.join(self.tags)}"
        )


class KnowledgeBase:
    def __init__(self):
        self.entries: list[KnowledgeEntry] = []
        self._embeddings: Optional[np.ndarray] = None

    def load(self) -> None:
        with open(KB_PATH, encoding="utf-8") as f:
            data = json.load(f)
        self.entries = [KnowledgeEntry(**item) for item in data]
        self._embeddings = embed_batch([e.to_search_text() for e in self.entries])
        print(f"[KnowledgeBase] Loaded {len(self.entries)} entries.")

    def search(self, query: str, top_k: int = 3) -> list[tuple[KnowledgeEntry, float]]:
        """Semantic search over KB entries. Returns (entry, score) pairs, best first."""
        if self._embeddings is None or len(self.entries) == 0:
            return []
        query_vec = embed_text(query)
        scores = self._embeddings @ query_vec  # cosine similarity (vectors are normalized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.entries[i], float(scores[i])) for i in top_indices]

    def get_by_id(self, entry_id: str) -> Optional[KnowledgeEntry]:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    def get_all(self) -> list[KnowledgeEntry]:
        return list(self.entries)


# Singleton
_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
        _kb.load()
    return _kb