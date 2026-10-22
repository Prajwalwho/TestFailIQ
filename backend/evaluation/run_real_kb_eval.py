"""
Knowledge-base retrieval on REAL labeled logs.

Each hand-verified real failure is a query; KB entries carry a coarse 'category'. A hit means the
top-ranked KB entries include one with the same category as the log's label. Real failures labeled
'other' (no category) act as the no-match set for threshold tuning.

Caveat: the KB entries were written from general CI knowledge plus patterns seen while labeling,
so these numbers are somewhat optimistic.

Run from the backend folder:  python evaluation/run_real_kb_eval.py
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from rank_bm25 import BM25Okapi
from evaluation.run_real_eval import load_documents, summarize, tokenize
from ingestion.log_parser import VARIANTS
from rag.retriever import KB_MIN_SCORE
from storage.knowledge_base import get_kb

OUT_PATH = os.path.join(BACKEND_DIR, "data", "eval_real_kb_results.json")
SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def category_ranking(ranked_entries):
    """Collapse an entry ranking to distinct categories, keeping each category's best score."""
    seen, out = set(), []
    for entry, score in ranked_entries:
        if entry.category and entry.category not in seen:
            seen.add(entry.category)
            out.append((entry.category, float(score)))
    return out


def at_threshold(positives, negatives, t):
    n_pos, n_neg = max(len(positives), 1), max(len(negatives), 1)
    correct = sum(1 for gold, r in positives if r and r[0][1] >= t and r[0][0] == gold)
    wrong = sum(1 for gold, r in positives if r and r[0][1] >= t and r[0][0] != gold)
    false_match = sum(1 for r in negatives if r and r[0][1] >= t)
    return {
        "threshold": t,
        "correct": round(correct / n_pos, 3),
        "wrong": round(wrong / n_pos, 3),
        "unknown_false_match": round(false_match / n_neg, 3),
    }


def main():
    kb = get_kb()
    entries = kb.get_all()
    if not any(e.category for e in entries):
        print("KB entries have no 'category'. Replace data/knowledge_base.json and storage/knowledge_base.py first.")
        return

    docs = load_documents()
    queries = [d for d in docs if d["label"] != "other"]
    unknowns = [d for d in docs if d["label"] == "other"]
    if len(queries) < 10:
        print(f"Only {len(queries)} usable labeled logs; label more before evaluating.")
        return

    counts = Counter(d["label"] for d in queries)
    print(f"{len(entries)} KB entries, {len(queries)} labeled real failures, "
          f"{len(unknowns)} 'other' failures used as the unknown set.")
    for label, count in counts.most_common():
        print(f"   {label:<24}{count:>3}")

    results, sweeps = {}, {}
    for variant in VARIANTS:
        positives, outcome = [], []
        for d in queries:
            ranking = category_ranking(kb.search(d["parsed"].to_search_text(variant), top_k=len(entries)))
            positives.append((d["label"], ranking))
            outcome.append((d["label"], [c for c, _ in ranking]))
        negatives = [category_ranking(kb.search(d["parsed"].to_search_text(variant), top_k=len(entries)))
                     for d in unknowns]
        results[f"embedding / {variant}"] = summarize(outcome)
        sweeps[variant] = [at_threshold(positives, negatives, t) for t in SWEEP]

    bm25 = BM25Okapi([tokenize(e.to_search_text()) for e in entries])
    outcome = []
    for d in queries:
        scores = np.asarray(bm25.get_scores(tokenize(d["parsed"].to_search_text("no_names"))))
        order = np.argsort(-scores, kind="stable")
        ranking = category_ranking([(entries[i], scores[i]) for i in order])
        outcome.append((d["label"], [c for c, _ in ranking]))
    results["bm25 / no_names"] = summarize(outcome)

    majority_order = [c for c, _ in counts.most_common()]
    results["majority class (baseline)"] = summarize([(d["label"], majority_order) for d in queries])

    print(f"\n{'system':<30}{'Hit@1 (95% CI)':>22}{'Hit@3':>8}{'MRR':>7}")
    for name, r in results.items():
        ci = f"{r['hit_at_1']:.0%} ({r['hit1_ci'][0]:.0%}-{r['hit1_ci'][1]:.0%})"
        print(f"{name:<30}{ci:>22}{r['hit_at_3']:>8.0%}{r['mrr']:>7.2f}")

    variant_keys = [f"embedding / {v}" for v in VARIANTS]
    best_key = max(variant_keys, key=lambda k: results[k]["hit1_count"])
    best = best_key.split(" / ")[1]
    print(f"\nPer-class Hit@1 for {best_key}:")
    for label, s in results[best_key]["per_label"].items():
        print(f"   {label:<24}{s['hits']}/{s['total']}")

    print(f"\nThreshold sweep for '{best}' (current KB_MIN_SCORE = {KB_MIN_SCORE}):")
    print("threshold | correct | wrong | unknown logs wrongly matched")
    for s in sweeps[best]:
        print(f"   {s['threshold']:.2f}   |  {s['correct']:>4.0%}   | {s['wrong']:>4.0%}  | {s['unknown_false_match']:>4.0%}")
    suggested = max(sweeps[best], key=lambda s: (s["correct"] - s["wrong"] - s["unknown_false_match"], -s["threshold"]))
    print(f"Suggested KB threshold: {suggested['threshold']}  (small dataset: treat as provisional)")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "kb_entries": len(entries), "queries": len(queries), "unknown_set": len(unknowns),
            "systems": results, "sweeps": sweeps, "suggested_threshold": suggested["threshold"],
        }, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()