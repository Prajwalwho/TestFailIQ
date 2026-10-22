"""
Derive the relevance thresholds and a confidence calibration from the real labeled logs and write
data/thresholds.json, which rag/retriever.py and rag/confidence.py load.

Every labeled real failure is a query. The top-1 result is "answered" when its similarity is >= the
threshold. Failures labeled 'other' are unknowns: answering them is always a mistake.

    precision = correct / (correct + wrong + answered unknowns)
    recall    = correct / labeled failures

Policy:
  * similar_min   : lowest threshold whose F0.5 is within 0.01 of the best (avoids a knife-edge peak)
  * kb_prompt_min : best F0.5 among thresholds with >= 10 answers and >= 40% precision. If none
                    qualifies the KB is NOT given to the LLM (threshold 1.01).
  * kb_display_min: same as kb_prompt_min (the UI should not show matches we would not trust)
  * calibration   : precision by similarity, made monotone with isotonic regression (PAV), used to
                    turn a similarity into a confidence

Run from the backend folder:  python evaluation/tune_thresholds.py
Re-run after changing the parser variant, the knowledge base or the labeled data, then restart the server.
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
TOLERANCE = 0.01
MIN_ANSWERS_SIMILAR = 15
MIN_ANSWERS_KB = 10
KB_PRECISION_FLOOR = 0.40
DISABLED = 1.01


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


def pick_similar(rows):
    eligible = [r for r in rows if r["answered"] >= MIN_ANSWERS_SIMILAR] or rows
    top = max(r["f05"] for r in eligible)
    near = [r for r in eligible if r["f05"] >= top - TOLERANCE]
    return min(near, key=lambda r: r["threshold"])


def pick_kb(rows):
    ok = [r for r in rows if r["answered"] >= MIN_ANSWERS_KB and r["precision"] >= KB_PRECISION_FLOOR]
    if not ok:
        return None
    top = max(r["f05"] for r in ok)
    return min((r for r in ok if r["f05"] == top), key=lambda r: r["threshold"])


def isotonic(values, weights):
    """Pool-adjacent-violators: the closest non-decreasing sequence to values."""
    blocks = []  # [mean, weight, count]
    for v, w in zip(values, weights):
        blocks.append([v, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out = []
    for mean, _, count in blocks:
        out.extend([mean] * count)
    return out


def calibration_table(rows):
    used = [r for r in rows if r["answered"] > 0]
    smoothed = isotonic([r["precision"] for r in used], [r["answered"] for r in used])
    return [{"threshold": r["threshold"], "precision": round(p, 3), "answered": r["answered"]}
            for r, p in zip(used, smoothed)]


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
    sim_t = pick_similar(sim_rows)
    kb_t = pick_kb(kb_rows)

    print_table("Similar past failures", sim_rows, {sim_t["threshold"]: ["similar_min"]})
    kb_marks = {kb_t["threshold"]: ["kb_display_min, kb_prompt_min"]} if kb_t else {}
    print_table("Knowledge base", kb_rows, kb_marks)

    kb_value = kb_t["threshold"] if kb_t else DISABLED
    calibration = calibration_table(sim_rows)
    print(f"\nChosen: similar_min={sim_t['threshold']}  kb_display_min={kb_value}  kb_prompt_min={kb_value}")
    print(f"   similar: {sim_t['answered']} answers, {sim_t['precision']:.0%} correct, "
          f"{sim_t['recall']:.0%} of labeled failures answered")
    if kb_t:
        print(f"   knowledge base: {kb_t['answered']} answers, {kb_t['precision']:.0%} correct, "
              f"{kb_t['recall']:.0%} of labeled failures answered")
    else:
        print(f"   knowledge base: no threshold reaches {KB_PRECISION_FLOOR:.0%} precision with "
              f"{MIN_ANSWERS_KB}+ answers, so KB entries are NOT given to the LLM.")
    print("Confidence calibration (similarity -> share of correct top matches):")
    for row in calibration:
        print(f"   >= {row['threshold']:.3f}: {row['precision']:.0%}  ({row['answered']} answers)")
    print("Small dataset: treat thresholds and calibration as provisional.")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "variant": DEFAULT_VARIANT,
            "similar_min": sim_t["threshold"],
            "kb_display_min": kb_value,
            "kb_prompt_min": kb_value,
            "calibration": calibration,
            "data": {"labeled_failures": len(labeled), "unknown_failures": len(unknown)},
            "at_chosen": {"similar": rounded(sim_t), "kb": rounded(kb_t) if kb_t else None},
        }, f, indent=2)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()