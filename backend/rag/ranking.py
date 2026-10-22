"""Pure ranking helpers (no model imports, so they are cheap to unit-test)."""


def rerank(scored, bonuses, min_score, limit):
    """
    scored:  [(item, raw_score)] where item has an `id`.
    bonuses: {id: bonus} derived from user feedback.

    Items are filtered on the RAW score, so feedback can never promote something below the relevance
    threshold. Surviving items are ordered by raw score plus bonus; the raw score is returned unchanged.
    """
    kept = [(item, score) for item, score in scored if score >= min_score]
    kept.sort(key=lambda pair: pair[1] + bonuses.get(pair[0].id, 0.0), reverse=True)
    return kept[:limit]