"""
Turns user feedback (thumbs up/down on diagnoses) into small ranking bonuses.

A diagnosis stores which past failure and which KB entry it relied on. Each vote moves that item's
bonus by STEP, capped at +/-CAP. The bonus only re-orders results that already passed the relevance
threshold (see rag/ranking.py); it never creates evidence.

Limitation: a thumbs-down cannot say whether the match, the KB entry or the LLM was at fault, so
both items on that diagnosis are penalised.
"""

from dataclasses import dataclass, field

from storage.diagnosis_store import get_diagnosis_store

STEP = 0.03
CAP = 0.09


@dataclass
class Adjustments:
    failures: dict = field(default_factory=dict)  # past failure id -> bonus
    kb: dict = field(default_factory=dict)        # KB entry id -> bonus


def _bonus(votes: int) -> float:
    return max(-CAP, min(CAP, STEP * votes))


def compute_adjustments(diagnoses) -> Adjustments:
    failure_votes, kb_votes = {}, {}
    for d in diagnoses:
        if d.feedback not in ("correct", "incorrect"):
            continue
        sign = 1 if d.feedback == "correct" else -1
        if d.top_match_id:
            failure_votes[d.top_match_id] = failure_votes.get(d.top_match_id, 0) + sign
        if d.kb_entry_used:
            kb_votes[d.kb_entry_used] = kb_votes.get(d.kb_entry_used, 0) + sign
    return Adjustments(
        failures={k: _bonus(v) for k, v in failure_votes.items()},
        kb={k: _bonus(v) for k, v in kb_votes.items()},
    )


def get_adjustments() -> Adjustments:
    try:
        return compute_adjustments(get_diagnosis_store().get_all())
    except Exception:  # feedback must never break retrieval
        return Adjustments()