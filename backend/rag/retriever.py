"""
RAG Retriever — combines FAISS failure search + KB search.
Returns the best evidence package for LLM generation.

Relevance thresholds come from data/thresholds.json (written by evaluation/tune_thresholds.py);
defaults apply when the file is missing. User feedback re-orders results that already passed the
threshold (rag/ranking.py, storage/feedback_signal.py).
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

from ingestion.log_parser import ParsedLog
from rag.ranking import rerank
from storage.faiss_store import FailureRecord, get_store
from storage.feedback_signal import get_adjustments
from storage.knowledge_base import KnowledgeEntry, get_kb

THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "thresholds.json")
_DEFAULTS = {"similar_min": 0.60, "kb_display_min": 0.50, "kb_prompt_min": 0.50}


def load_thresholds() -> dict:
    values = dict(_DEFAULTS)
    try:
        with open(THRESHOLDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for key in values:
            if key in data:
                values[key] = float(data[key])
    except (OSError, ValueError, TypeError):
        pass
    return values


_T = load_thresholds()
SIMILAR_MIN_SCORE = _T["similar_min"]                           # past failures shown and given to the LLM
KB_MIN_SCORE = _T["kb_display_min"]                             # KB cards shown in the UI
KB_PROMPT_MIN_SCORE = max(_T["kb_prompt_min"], KB_MIN_SCORE)    # KB entry given to the LLM as evidence
print(f"[Retriever] thresholds: similar>={SIMILAR_MIN_SCORE} kb_display>={KB_MIN_SCORE} "
      f"kb_prompt>={KB_PROMPT_MIN_SCORE}")


@dataclass
class RetrievedEvidence:
    similar_failures: list[tuple[FailureRecord, float]]   # (record, raw similarity)
    kb_matches: list[tuple[KnowledgeEntry, float]]         # (entry, raw similarity), shown in the UI
    top_kb_entry: Optional[KnowledgeEntry]                 # only set when strong enough for the prompt
    top_similarity: float                                   # best raw similarity among kept failures


def retrieve(parsed_log: ParsedLog, top_k: int = 5, kb_top_k: int = 3) -> RetrievedEvidence:
    """
    Given a parsed log, retrieve:
    1. Top-k similar historical failures from FAISS
    2. Top-k matching KB entries via semantic search
    Both lists are thresholded on the raw score, then ordered with feedback bonuses.
    """
    search_text = parsed_log.to_search_text()
    adjustments = get_adjustments()

    similar_failures = rerank(
        get_store().search(search_text, top_k=top_k * 3), adjustments.failures, SIMILAR_MIN_SCORE, top_k)
    kb_matches = rerank(
        get_kb().search(search_text, top_k=kb_top_k * 3), adjustments.kb, KB_MIN_SCORE, kb_top_k)

    top_kb_entry = kb_matches[0][0] if kb_matches and kb_matches[0][1] >= KB_PROMPT_MIN_SCORE else None
    top_similarity = max((score for _, score in similar_failures), default=0.0)

    return RetrievedEvidence(
        similar_failures=similar_failures,
        kb_matches=kb_matches,
        top_kb_entry=top_kb_entry,
        top_similarity=top_similarity,
    )