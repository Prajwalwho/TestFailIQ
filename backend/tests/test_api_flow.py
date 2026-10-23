from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
import storage.diagnosis_store as ds
from rag.generator import DiagnosisResult
from rag.retriever import RetrievedEvidence
from storage.faiss_store import FailureRecord

LOG = "FAILED tests/test_cart.py::test_total - AssertionError: assert 10 == 11"

RECORD = FailureRecord(
    id="LOG-9", test_name="test_total", module="cart", error_type="AssertionError",
    error_message="assert 10 == 12", stacktrace="", log_text="x", timestamp="2026-01-01T00:00:00",
    resolution="hand-labeled category: assertion_failure",
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "STORE_PATH", str(tmp_path / "diagnoses.json"))
    monkeypatch.setattr(ds, "_store", ds.DiagnosisStore())
    evidence = RetrievedEvidence(
        similar_failures=[(RECORD, 0.71)], kb_matches=[], top_kb_entry=None, top_similarity=0.71)
    monkeypatch.setattr(main, "retrieve", lambda parsed, top_k=5: evidence)
    monkeypatch.setattr(main, "generate_diagnosis", lambda parsed, ev: DiagnosisResult(
        root_cause="Assertion differs", confidence=0.56, explanation="Expected and actual differ.",
        next_steps=["Check the expected value", "Re-run the test"], fix_snippet=None,
        model_used="fake-llm", similar_failures_used=1, kb_entry_used=None))
    return TestClient(main.app)  # no `with`: startup (production index) is not triggered


def test_diagnose_returns_structured_result(client):
    response = client.post("/api/diagnose", json={"log_text": LOG})
    assert response.status_code == 200
    body = response.json()
    assert body["diagnosis_id"].startswith("DX-")
    assert body["parsed_log"]["error_type"] == "AssertionError"
    assert body["diagnosis"]["model_used"] == "fake-llm"
    assert body["evidence"]["similar_failures"][0]["id"] == "LOG-9"
    assert body["evidence"]["kb_matches"] == []


def test_feedback_updates_the_stored_diagnosis(client):
    diagnosis_id = client.post("/api/diagnose", json={"log_text": LOG}).json()["diagnosis_id"]
    response = client.post("/api/feedback", json={"diagnosis_id": diagnosis_id, "feedback": "correct"})
    assert response.status_code == 200
    assert ds.get_diagnosis_store().get_all()[0].feedback == "correct"


def test_feedback_rejects_unknown_value(client):
    diagnosis_id = client.post("/api/diagnose", json={"log_text": LOG}).json()["diagnosis_id"]
    response = client.post("/api/feedback", json={"diagnosis_id": diagnosis_id, "feedback": "maybe"})
    assert response.status_code == 400


def test_feedback_for_unknown_diagnosis_is_404(client):
    response = client.post("/api/feedback", json={"diagnosis_id": "DX-NOPE", "feedback": "correct"})
    assert response.status_code == 404


@pytest.mark.parametrize("payload", [
    {"log_text": ""},
    {"log_text": "x" * (main.MAX_LOG_CHARS + 1)},
    {"log_text": LOG, "top_k": 0},
    {"log_text": LOG, "top_k": 21},
])
def test_diagnose_validates_input(client, payload):
    assert client.post("/api/diagnose", json=payload).status_code == 422


def test_metrics_counts_feedback(client, monkeypatch):
    monkeypatch.setattr(main, "get_store", lambda: SimpleNamespace(size=7))
    first = client.post("/api/diagnose", json={"log_text": LOG}).json()["diagnosis_id"]
    client.post("/api/diagnose", json={"log_text": LOG})
    client.post("/api/feedback", json={"diagnosis_id": first, "feedback": "incorrect"})
    body = client.get("/api/metrics").json()
    assert body["total_indexed"] == 7
    assert body["total_diagnoses"] == 2
    assert body["feedback_incorrect"] == 1
    assert body["feedback_pending"] == 1
    assert body["resolution_rate"] == 0.0