"""
Derive the relevance thresholds from the real labeled logs and write data/thresholds.json,
which rag/retriever.py loads at import time.

For each retrieval system (similar past failures, knowledge base) every real failure is used as a
query. The top-1 result is "answered" when its similarity is >= the threshold. Failures labeled
'other' are unknowns: answering them is always a mistake.

    precision = correct / (correct + wrong + answered unknowns)
    recall    = correct / labeled failures

F0.5 (precision-weighted) picks what is given to the LLM as evidence; F1 picks the knowledge-base
display threshold. Re-run after changing the parser variant, the knowledge base or the labeled data,
then restart the server.

Run from the backend folder:  python evaluation/tune_thresholds.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from evaluation.run_real_eval import allowed_indices, load_documents
from evaluation.run_real_kb_eval import category_ranking
from ingestion.embedder import embed_batch
from ingestion.log_parser import DEFAULT_VARIANT
from storage.knowledge_base import get_kb

OUT_PATH = os.path.join(BACKEND_DIR, "data", "thresholds.json")
SWEEP = [round(0.30 + 0.025 * i, 3) for i in range(21)]  # 0.30 .. 0.80


def f_beta(precision, recall, beta):
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def sweep(pos, neg):
    """pos: [(is_correct, score)] for labeled failures; neg: [score] for unknown failures."""
    rows = []
    for t in SWEEP:
        correct = sum(1 for ok, s in pos if s >= t and ok)
        wrong = sum(1 for ok, s in pos if s >= t and not ok)
        unknown = sum(1 for s in neg if s >= t)
        answered = correct + wrong + unknown
        precision = correct / answered if answered else 0.0
        recall = correct / len(pos) if pos else 0.0
        rows.append({
            "threshold": t, "answered": answered, "precision": precision, "recall": recall,
            "f1": f_beta(precision, recall, 1.0), "f05": f_beta(precision, recall, 0.5),
        })
    return rows


def best(rows, key):
    """Highest score; ties go to the lowest threshold."""
    top = max(r[key] for r in rows)
    return min((r for r in rows if r[key] == top), key=lambda r: r["threshold"])


def rounded(row):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}


def print_table(title, rows, marks):
    print(f"\n{title}")
    print("threshold | answered | precision | recall |   F1  | F0.5")
    for r in rows:
        flag = "  <-- " + ", ".join(marks[r["threshold"]]) if r["threshold"] in marks else ""
        print(f"   {r['threshold']:.3f}  |   {r['answered']:>3}    |   {r['precision']:>4.0%}    |  {r['recall']:>4.0%}  "
              f"| {r['f1']:.2f} | {r['f05']:.2f}{flag}")


def main():
    docs = load_documents()
    labeled = [i for i, d in enumerate(docs) if d["label"] != "other"]
    unknown = [i for i, d in enumerate(docs) if d["label"] == "other"]
    if len(labeled) < 20:
        print(f"Only {len(labeled)} labeled real failures; label more before tuning.")
        return
    print(f"Search text variant: {DEFAULT_VARIANT}. {len(labeled)} labeled failures, {len(unknown)} unknowns.")

    texts = [d["parsed"].to_search_text() for d in docs]
    E = embed_batch(texts)

    # Similar past failures: leave-one-run-out, top-1 neighbour
    sim_pos, sim_neg = [], []
    for q in labeled + unknown:
        allowed = allowed_indices(docs, q)
        if len(allowed) == 0:
            continue
        scores = E[allowed] @ E[q]
        j = int(allowed[int(np.argmax(scores))])
        score = float(np.max(scores))
        if docs[q]["label"] == "other":
            sim_neg.append(score)
        else:
            sim_pos.append((docs[j]["label"] == docs[q]["label"], score))

    # Knowledge base: top-1 category
    kb = get_kb()
    n_entries = len(kb.get_all())
    kb_pos, kb_neg = [], []
    for q in labeled + unknown:
        ranking = category_ranking(kb.search(texts[q], top_k=n_entries))
        if not ranking:
            continue
        category, score = ranking[0]
        if docs[q]["label"] == "other":
            kb_neg.append(score)
        else:
            kb_pos.append((category == docs[q]["label"], score))

    sim_rows, kb_rows = sweep(sim_pos, sim_neg), sweep(kb_pos, kb_neg)
    sim_t = best(sim_rows, "f05")
    kb_display = best(kb_rows, "f1")
    kb_prompt = best(kb_rows, "f05")
    if kb_prompt["threshold"] < kb_display["threshold"]:
        kb_prompt = kb_display

    print_table("Similar past failures (what goes to the LLM and the UI list)", sim_rows,
                {sim_t["threshold"]: ["similar_min (F0.5)"]})
    marks = {}
    marks.setdefault(kb_display["threshold"], []).append("kb_display_min (F1)")
    marks.setdefault(kb_prompt["threshold"], []).append("kb_prompt_min (F0.5)")
    print_table("Knowledge base", kb_rows, marks)

    print(f"\nChosen: similar_min={sim_t['threshold']}  kb_display_min={kb_display['threshold']}  "
          f"kb_prompt_min={kb_prompt['threshold']}")
    for name, row in (("similar_min", sim_t), ("kb_prompt_min", kb_prompt)):
        print(f"   at {name}: {row['answered']} answers, {row['precision']:.0%} of them correct, "
              f"{row['recall']:.0%} of labeled failures answered")
        if row["precision"] < 0.5:
            print("   WARNING: fewer than half of the answers are right even at the best threshold; "
                  "treat these matches as hints, not facts.")
    print("Small dataset: treat thresholds as provisional until more real logs are labeled.")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "variant": DEFAULT_VARIANT,
            "similar_min": sim_t["threshold"],
            "kb_display_min": kb_display["threshold"],
            "kb_prompt_min": kb_prompt["threshold"],
            "data": {"labeled_failures": len(labeled), "unknown_failures": len(unknown)},
            "at_chosen": {"similar": rounded(sim_t), "kb_display": rounded(kb_display), "kb_prompt": rounded(kb_prompt)},
        }, f, indent=2)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()