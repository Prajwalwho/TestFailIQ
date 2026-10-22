import json
import os

import pytest
from fastapi.testclient import TestClient

import main

# No `with` block: the startup handler (which loads the production index) is not triggered.
client = TestClient(main.app)


def test_eval_results_route_is_registered():
    assert "/api/eval-results" in {route.path for route in main.app.routes}


def test_missing_file_returns_404_with_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "EVAL_RESULTS_PATH", str(tmp_path / "missing.json"))
    response = client.get("/api/eval-results")
    assert response.status_code == 404
    assert "run_eval.py" in response.json()["detail"]


def test_valid_file_is_returned_unchanged(tmp_path, monkeypatch):
    payload = {"index_size": 30, "test_sets": {"standard": {}, "hard": {}, "combined": {}}}
    path = tmp_path / "eval_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(main, "EVAL_RESULTS_PATH", str(path))
    response = client.get("/api/eval-results")
    assert response.status_code == 200
    assert response.json() == payload


def test_corrupt_file_returns_500_not_a_crash(tmp_path, monkeypatch):
    path = tmp_path / "eval_results.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(main, "EVAL_RESULTS_PATH", str(path))
    response = client.get("/api/eval-results")
    assert response.status_code == 500
    assert "could not be read" in response.json()["detail"]


def test_real_results_file_matches_what_the_panel_needs():
    """Contract with frontend/src/components/HeldOutEval.jsx (skipped if the file does not exist yet)."""
    if not os.path.exists(main.EVAL_RESULTS_PATH):
        pytest.skip("data/eval_results.json not generated yet")
    with open(main.EVAL_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data["index_size"], int)
    for set_name in ("standard", "hard", "combined"):
        test_set = data["test_sets"][set_name]
        assert test_set["systems"], set_name
        for name, system in test_set["systems"].items():
            for key in ("hit_at_1", "hit_at_3", "mrr", "separation_auc", "n_positive", "n_negative", "misses"):
                assert key in system, f"{set_name}/{name} lacks {key}"
    for name in ("failures_embedding", "kb_embedding"):
        system = data["test_sets"]["combined"]["systems"][name]
        for key in ("sweep", "at_current_threshold", "current_threshold", "suggested_threshold"):
            assert key in system, f"combined/{name} lacks {key}"