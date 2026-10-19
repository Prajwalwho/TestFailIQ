"""
Held-out retrieval evaluation.

Builds a TEMPORARY in-memory index from the eval set's "index" split, queries it with
the "test" split, and scores results against the labels (known_issue_id).
Never touches the production index on disk.

Run from the backend folder:  python evaluation/run_eval.py
"""

import json
import os
import sys
from datetime import datetime

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from ingestion.log_parser import parse_log
from storage.faiss_store import FAISSStore, FailureRecord
from storage.knowledge_base import get_kb
from rag.retriever import SIMILAR_MIN_SCORE, KB_MIN_SCORE

EVAL_SET_PATH = os.path.join(BACKEND_DIR, "data", "eval_set.json")
RESULTS_PATH = os.path.join(BACKEND_DIR, "data", "eval_results.json")
TOP_K = 5
SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


# ── Building the temporary index ─────────────────────────────────────────────
def build_index(entries) -> FAISSStore:
    store = FAISSStore()  # fresh, in-memory; we never call save()
    for e in entries:
        if e["split"] != "index":
            continue
        p = parse_log(e["log_text"])
        record = FailureRecord(
            id=e["id"],
            test_name=p.test_name,
            module=p.module,
            error_type=p.error_type,
            error_message=p.error_message,
            stacktrace=p.stacktrace,
            log_text=e["log_text"],
            timestamp=datetime.utcnow().isoformat(),
            known_issue_id=e["known_issue_id"],
            resolution=None,
        )
        store.add(record, p.to_search_text())
    return store


# ── Retrieval systems: query text -> ranked [(label, score), ...] ────────────
def make_failure_retriever(store):
    def retrieve(query):
        return [(rec.known_issue_id, score) for rec, score in store.search(query, top_k=TOP_K)]
    return retrieve


def make_kb_retriever(kb):
    def retrieve(query):
        return [(entry.id, score) for entry, score in kb.search(query, top_k=TOP_K)]
    return retrieve


# ── Scoring ──────────────────────────────────────────────────────────────────
def reciprocal_rank(ranked, gold):
    for i, (label, _) in enumerate(ranked, 1):
        if label == gold:
            return 1.0 / i
    return 0.0


def at_threshold(rows, t):
    """rows: list of (gold, ranked). Answer only when the top-1 score >= t."""
    pos = [(g, r) for g, r in rows if g]
    neg = [(g, r) for g, r in rows if not g]
    correct = sum(1 for g, r in pos if r and r[0][1] >= t and r[0][0] == g)
    wrong = sum(1 for g, r in pos if r and r[0][1] >= t and r[0][0] != g)
    neg_fp = sum(1 for g, r in neg if r and r[0][1] >= t)
    return {
        "threshold": t,
        "recall": round(correct / len(pos), 3) if pos else 0.0,
        "wrong": round(wrong / len(pos), 3) if pos else 0.0,
        "negative_false_match": round(neg_fp / len(neg), 3) if neg else 0.0,
    }


def evaluate(retrieve, queries, current_threshold):
    rows = [(q["gold"], retrieve(q["text"])) for q in queries]
    pos = [(q, r) for q, (g, r) in zip(queries, rows) if q["gold"]]
    neg = [(q, r) for q, (g, r) in zip(queries, rows) if not q["gold"]]

    hit1 = hit3 = 0
    rr_sum = 0.0
    per_issue, misses = {}, []
    for q, ranked in pos:
        gold = q["gold"]
        labels = [label for label, _ in ranked]
        ok1 = bool(ranked) and labels[0] == gold
        ok3 = gold in labels[:3]
        hit1 += ok1
        hit3 += ok3
        rr_sum += reciprocal_rank(ranked, gold)
        stats = per_issue.setdefault(gold, {"hits": 0, "total": 0})
        stats["total"] += 1
        stats["hits"] += int(ok1)
        if not ok1:
            misses.append({
                "id": q["id"],
                "gold": gold,
                "predicted": labels[0] if labels else None,
                "top1_score": round(ranked[0][1], 3) if ranked else 0.0,
                "gold_rank": labels.index(gold) + 1 if gold in labels else None,
            })

    n = len(pos)
    top1_pos = [r[0][1] for _, r in pos if r]
    top1_neg = [r[0][1] for _, r in neg if r]
    sweep = [at_threshold(rows, t) for t in SWEEP]

    # Suggested threshold: best (recall - wrong - false-match); ties go to the lower threshold
    best = max(sweep, key=lambda s: (s["recall"] - s["wrong"] - s["negative_false_match"], -s["threshold"]))

    return {
        "n_positive": n,
        "n_negative": len(neg),
        "hit_at_1": round(hit1 / n, 3) if n else 0.0,
        "hit_at_3": round(hit3 / n, 3) if n else 0.0,
        "mrr": round(rr_sum / n, 3) if n else 0.0,
        "mean_top1_score_positive": round(sum(top1_pos) / len(top1_pos), 3) if top1_pos else 0.0,
        "mean_top1_score_negative": round(sum(top1_neg) / len(top1_neg), 3) if top1_neg else 0.0,
        "current_threshold": current_threshold,
        "at_current_threshold": at_threshold(rows, current_threshold),
        "suggested_threshold": best["threshold"],
        "sweep": sweep,
        "per_issue": per_issue,
        "misses": misses,
    }


def print_report(name, res, index_size):
    print(f"\n=== {name}  (index={index_size}, {res['n_positive']} labeled + {res['n_negative']} no-match test logs) ===")
    print(f"Hit@1 {res['hit_at_1']:.0%} | Hit@3 {res['hit_at_3']:.0%} | MRR {res['mrr']:.3f}")
    print(f"Mean top-1 score: matches {res['mean_top1_score_positive']:.2f} vs no-match {res['mean_top1_score_negative']:.2f}")
    cur = res["at_current_threshold"]
    print(f"At current threshold {cur['threshold']}: correct {cur['recall']:.0%}, "
          f"wrong {cur['wrong']:.0%}, no-match logs wrongly matched {cur['negative_false_match']:.0%}")
    print("threshold | correct | wrong | no-match false match")
    for s in res["sweep"]:
        print(f"   {s['threshold']:.2f}   |  {s['recall']:>4.0%}   | {s['wrong']:>4.0%}  | {s['negative_false_match']:>4.0%}")
    print(f"Suggested threshold: {res['suggested_threshold']}")
    if res["misses"]:
        print("Top-1 misses:")
        for m in res["misses"]:
            print(f"   {m['id']}: expected {m['gold']}, got {m['predicted']} "
                  f"(score {m['top1_score']}, correct answer ranked #{m['gold_rank']})")


def main():
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    # Leak check: no test log may share its text or id with an index log
    index_entries = [e for e in entries if e["split"] == "index"]
    test_entries = [e for e in entries if e["split"] == "test"]
    assert not {e["id"] for e in index_entries} & {e["id"] for e in test_entries}, "id overlap"
    assert not {e["log_text"] for e in index_entries} & {e["log_text"] for e in test_entries}, "text overlap"
    print("Leak check passed: index and test splits share no logs.")

    store = build_index(entries)
    kb = get_kb()

    queries = []
    for e in test_entries:
        p = parse_log(e["log_text"])
        queries.append({"id": e["id"], "gold": e["known_issue_id"], "text": p.to_search_text()})

    systems = {
        "failure_retrieval": (make_failure_retriever(store), SIMILAR_MIN_SCORE),
        "kb_retrieval": (make_kb_retriever(kb), KB_MIN_SCORE),
    }
    results = {
        "generated_at": datetime.utcnow().isoformat(),
        "index_size": store.size,
        "test_size": len(queries),
        "systems": {},
    }
    for name, (fn, threshold) in systems.items():
        res = evaluate(fn, queries, threshold)
        results["systems"][name] = res
        print_report(name, res, store.size)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()