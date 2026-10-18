"""
Evaluation metrics for the RAG retrieval and diagnosis pipeline.
"""

from dataclasses import dataclass, field
from storage.diagnosis_store import get_diagnosis_store


@dataclass
class EvalMetrics:
    total_indexed: int = 0
    total_diagnoses: int = 0
    feedback_correct: int = 0
    feedback_incorrect: int = 0
    feedback_pending: int = 0
    avg_similarity_score: float = 0.0
    resolution_rate: float = 0.0   # correct / (correct + incorrect)
    retrieval_at_1: float = 0.0    # % of queries where top-1 is same error type
    retrieval_at_3: float = 0.0


def compute_metrics() -> EvalMetrics:
    """Compute evaluation metrics from the current FAISS store state."""
    store = get_store()
    records = store.get_all()

    total = len(records)
    diagnoses = get_diagnosis_store().get_all()
    correct = sum(1 for d in diagnoses if d.feedback == "correct")
    incorrect = sum(1 for d in diagnoses if d.feedback == "incorrect")
    pending = sum(1 for d in diagnoses if d.feedback is None)

    resolution_rate = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0

    # Retrieval@k — evaluate by re-searching each record against itself
    # and checking if top-k contains the same error_type
    hits_at_1, hits_at_3 = 0, 0
    if total > 1:
        for record in records:
            search_text = (
                f"Test: {record.test_name}\n"
                f"Module: {record.module}\n"
                f"Error: {record.error_type}: {record.error_message}\n"
                f"Stacktrace:\n{record.stacktrace}"
            )
            results = store.search(search_text, top_k=3)
            # Filter out exact self-match
            results = [(r, s) for r, s in results if r.id != record.id]

            if results:
                top_1_error = results[0][0].error_type
                top_3_errors = [r.error_type for r, _ in results[:3]]
                if top_1_error == record.error_type:
                    hits_at_1 += 1
                if record.error_type in top_3_errors:
                    hits_at_3 += 1

    n = max(total - 1, 1)  # exclude self from denominator

    return EvalMetrics(
        total_indexed=total,
        total_diagnoses=correct + incorrect + pending,
        feedback_correct=correct,
        feedback_incorrect=incorrect,
        feedback_pending=pending,
        resolution_rate=round(resolution_rate, 3),
        retrieval_at_1=round(hits_at_1 / n, 3) if total > 1 else 0.0,
        retrieval_at_3=round(hits_at_3 / n, 3) if total > 1 else 0.0,
    )
