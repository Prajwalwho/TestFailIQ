"""
RAG Retriever — combines FAISS failure search + KB search.
Returns the best evidence package for LLM generation.
"""

from dataclasses import dataclass
from typing import Optional

from ingestion.log_parser import ParsedLog
from storage.faiss_store import FailureRecord, get_store
from storage.knowledge_base import KnowledgeEntry, get_kb


SIMILAR_MIN_SCORE = 0.50   # drop historical failures less similar than this
KB_MIN_SCORE = 0.45        # drop KB entries less similar than this

@dataclass
class RetrievedEvidence:
    similar_failures: list[tuple[FailureRecord, float]]   # (record, similarity_score)
    kb_matches: list[tuple[KnowledgeEntry, float]]         # (entry, similarity_score)
    top_kb_entry: Optional[KnowledgeEntry]
    top_similarity: float                                   # 0.0–1.0


def retrieve(parsed_log: ParsedLog, top_k: int = 5, kb_top_k: int = 3) -> RetrievedEvidence:
    """
    Given a parsed log, retrieve:
    1. Top-k similar historical failures from FAISS
    2. Top-k matching KB entries via semantic search
    """
    search_text = parsed_log.to_search_text()

    # --- FAISS search ---
    store = get_store()
    similar_failures = [
        (rec, score)
        for rec, score in store.search(search_text, top_k=top_k)
        if score >= SIMILAR_MIN_SCORE
    ]

    # --- KB semantic search ---
    kb = get_kb()
    kb_matches = [
        (entry, score)
        for entry, score in kb.search(search_text, top_k=kb_top_k)
        if score >= KB_MIN_SCORE
    ]

    top_kb_entry = kb_matches[0][0] if kb_matches else None
    top_similarity = similar_failures[0][1] if similar_failures else 0.0

    return RetrievedEvidence(
        similar_failures=similar_failures,
        kb_matches=kb_matches,
        top_kb_entry=top_kb_entry,
        top_similarity=top_similarity,
    )
