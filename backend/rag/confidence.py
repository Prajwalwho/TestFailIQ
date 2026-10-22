"""
Calibrated confidence: maps the best retrieval similarity to the share of top-1 matches that were
correct at that similarity on labeled real logs (the "calibration" table in data/thresholds.json,
written by evaluation/tune_thresholds.py).

This is a retrieval confidence: how often matches this similar were right, not a guarantee that the
generated diagnosis is right.
"""

import json
import os

THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "thresholds.json")
FLOOR, CEIL = 0.10, 0.95


def load_calibration() -> list[dict]:
    try:
        with open(THRESHOLDS_PATH, encoding="utf-8") as f:
            table = json.load(f).get("calibration", [])
        return sorted(table, key=lambda row: row["threshold"])
    except (OSError, ValueError, KeyError, TypeError):
        return []


_TABLE = load_calibration()


def confidence_from_similarity(similarity: float, table=None) -> float:
    table = _TABLE if table is None else table
    if similarity <= 0:
        return FLOOR  # no supporting evidence at all
    if not table:
        return round(min(max(similarity * 0.5, FLOOR), 0.5), 3)  # conservative default before calibration
    value = FLOOR
    for row in table:  # ascending thresholds
        if similarity >= row["threshold"]:
            value = row["precision"]
        else:
            break
    return round(min(max(value, FLOOR), CEIL), 3)