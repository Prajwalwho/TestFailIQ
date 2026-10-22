from types import SimpleNamespace

import pytest

from rag.confidence import confidence_from_similarity
from rag.ranking import rerank
from storage.feedback_signal import CAP, compute_adjustments


def item(item_id):
    return SimpleNamespace(id=item_id)


def diagnosis(feedback, top=None, kb=None):
    return SimpleNamespace(feedback=feedback, top_match_id=top, kb_entry_used=kb)


def test_rerank_orders_by_raw_score_without_feedback():
    out = rerank([(item("B"), 0.60), (item("A"), 0.80)], {}, 0.5, 5)
    assert [i.id for i, _ in out] == ["A", "B"]


def test_feedback_can_flip_close_scores_but_keeps_raw_score():
    out = rerank([(item("A"), 0.70), (item("B"), 0.68)], {"B": 0.03}, 0.5, 5)
    assert [i.id for i, _ in out] == ["B", "A"]
    assert out[0][1] == 0.68


def test_feedback_cannot_promote_items_below_threshold():
    out = rerank([(item("A"), 0.70), (item("B"), 0.40)], {"B": CAP}, 0.5, 5)
    assert [i.id for i, _ in out] == ["A"]


def test_rerank_respects_limit():
    scored = [(item(str(n)), 0.9 - n * 0.01) for n in range(6)]
    assert len(rerank(scored, {}, 0.5, 3)) == 3


def test_adjustments_from_votes():
    adj = compute_adjustments([
        diagnosis("correct", "LOG-1", "KB-1"),
        diagnosis("correct", "LOG-1"),
        diagnosis("incorrect", "LOG-2", "KB-1"),
    ])
    assert adj.failures["LOG-1"] == pytest.approx(0.06)
    assert adj.failures["LOG-2"] == pytest.approx(-0.03)
    assert adj.kb["KB-1"] == pytest.approx(0.0)


def test_adjustments_ignore_pending_feedback():
    adj = compute_adjustments([diagnosis(None, "LOG-1", "KB-1")])
    assert adj.failures == {} and adj.kb == {}


def test_adjustments_are_capped():
    adj = compute_adjustments([diagnosis("correct", "LOG-1") for _ in range(10)])
    assert adj.failures["LOG-1"] == pytest.approx(CAP)


def test_confidence_uses_calibration_table():
    table = [{"threshold": 0.6, "precision": 0.5}, {"threshold": 0.7, "precision": 0.6}]
    assert confidence_from_similarity(0.65, table) == 0.5
    assert confidence_from_similarity(0.75, table) == 0.6
    assert confidence_from_similarity(0.2, table) == 0.1


def test_confidence_floor_and_default_without_table():
    assert confidence_from_similarity(0.0, []) == 0.1
    assert confidence_from_similarity(0.8, []) == 0.4