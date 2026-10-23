"""
Persistent store for diagnoses (one record per /api/diagnose call).
Feedback is attached to a diagnosis, not to an indexed failure.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diagnoses.json")


@dataclass
class DiagnosisRecord:
    id: str
    timestamp: str
    test_name: str
    module: str
    error_type: str
    root_cause: str
    confidence: float
    model_used: str
    top_match_id: Optional[str] = None     # id of the top similar failure
    kb_entry_used: Optional[str] = None
    feedback: Optional[str] = None         # "correct" | "incorrect" | None


class DiagnosisStore:
    def __init__(self):
        self.records: list[DiagnosisRecord] = []
        self.load()

    def load(self) -> None:
        if os.path.exists(STORE_PATH):
            with open(STORE_PATH, encoding="utf-8") as f:
                self.records = [DiagnosisRecord(**item) for item in json.load(f)]

    def save(self) -> None:
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)

    def add(self, **fields) -> DiagnosisRecord:
        record = DiagnosisRecord(
            id=f"DX-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            **fields,
        )
        self.records.append(record)
        self.save()
        return record

    def update_feedback(self, diagnosis_id: str, feedback: str) -> bool:
        for r in self.records:
            if r.id == diagnosis_id:
                r.feedback = feedback
                self.save()
                return True
        return False

    def get_all(self) -> list[DiagnosisRecord]:
        return list(self.records)


_store: DiagnosisStore | None = None


def get_diagnosis_store() -> DiagnosisStore:
    global _store
    if _store is None:
        _store = DiagnosisStore()
    return _store