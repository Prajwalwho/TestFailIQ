"""
Retrieval evaluation on REAL labeled CI logs (leave-one-run-out).

For every labeled failure ("query"), the index contains all other labeled logs EXCEPT those
from the same CI run or with the same key error line, so near-duplicates cannot leak.
A hit means the top-ranked results include a log with the same coarse category.

Compares: five choices for which text is embedded, BM25 keywords, hybrid fusion and a
majority-class baseline. Reports 95% confidence intervals because the dataset is small.

Run from the backend folder:  python evaluation/run_real_eval.py
Only logs you verified yourself (reviewed = yes) are used.
"""

import csv
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from rank_bm25 import BM25Okapi
from ingestion.embedder import embed_batch
from ingestion.log_parser import VARIANTS, parse_log

DATA_DIR = os.path.join(BACKEND_DIR, "data")
REAL_DIR = os.path.join(DATA_DIR, "real_logs")
LOG_DIR = os.path.join(REAL_DIR, "logs")
CSV_PATH = os.path.join(REAL_DIR, "labels.csv")
MANIFEST_PATH = os.path.join(REAL_DIR, "manifest.json")
RESULTS_PATH = os.path.join(DATA_DIR, "eval_real_results.json")

# Fine labels are too thin for 80 logs, so they are merged into coarse classes.
COARSE = {
    "assertion_mismatch": "assertion_failure",
    "missing_dependency": "dependency_or_import",
    "build_or_compile": "build_or_compile",
    "null_or_type_error": "runtime_exception",
    "timeout": "environment_or_infra",
    "network_or_service": "environment_or_infra",
    "file_or_path": "environment_or_infra",
    "resource_exhaustion": "environment_or_infra",
    "config_or_env": "environment_or_infra",
    "flaky_race": "environment_or_infra",
}


def safe(text):
    return text.encode("ascii", "replace").decode()


def tokenize(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.lower())


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ── Data ─────────────────────────────────────────────────────────────────────
def load_documents():
    meta = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            meta = {m["id"]: m for m in json.load(f)}
    docs = []
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("reviewed") != "yes":
                continue
            category = row["category"].strip()
            if category in ("", "skip"):
                continue
            label = "other" if category == "other" else COARSE.get(category)
            path = os.path.join(LOG_DIR, f"{row['id']}.log")
            if label is None or not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as lf:
                text = lf.read()
            docs.append({
                "id": row["id"],
                "label": label,
                "fine": category,
                "key_line": row["key_line"],
                "run_id": meta.get(row["id"], {}).get("run_id"),
                "norm_key": re.sub(r"\d+", "N", row["key_line"]),
                "parsed": parse_log(text),
            })
    return docs


def allowed_indices(docs, q):
    d = docs[q]
    out = []
    for j, other in enumerate(docs):
        if j == q:
            continue
        if d["run_id"] is not None and other["run_id"] == d["run_id"]:
            continue
        if other["norm_key"] and other["norm_key"] == d["norm_key"]:
            continue
        out.append(j)
    return np.array(out, dtype=int)


def label_ranking(order, docs):
    seen, out = set(), []
    for j in order:
        label = docs[j]["label"]
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


# ── Retrieval systems: (query index, allowed doc indices) -> ranked labels ───
def embedding_system(E, docs):
    def fn(q, allowed):
        scores = E[allowed] @ E[q]
        return label_ranking(allowed[np.argsort(-scores, kind="stable")], docs)
    return fn


def bm25_system(tokens, docs):
    def fn(q, allowed):
        scores = np.asarray(BM25Okapi([tokens[j] for j in allowed]).get_scores(tokens[q]))
        return label_ranking(allowed[np.argsort(-scores, kind="stable")], docs)
    return fn


def hybrid_system(E, tokens, docs, k=60):
    def fn(q, allowed):
        emb_order = allowed[np.argsort(-(E[allowed] @ E[q]), kind="stable")]
        bm = np.asarray(BM25Okapi([tokens[j] for j in allowed]).get_scores(tokens[q]))
        bm_order = allowed[np.argsort(-bm, kind="stable")]
        fused = {}
        for order in (emb_order, bm_order):
            for rank, j in enumerate(order, 1):
                fused[j] = fused.get(j, 0.0) + 1.0 / (k + rank)
        return label_ranking(sorted(fused, key=lambda j: -fused[j]), docs)
    return fn


def majority_system(docs):
    """Chance baseline: always predict the most common labeled class ('other' is never a query label)."""
    order = [label for label, _ in Counter(d["label"] for d in docs if d["label"] != "other").most_common()]

    def fn(q, allowed):
        return order

    return fn


# ── Scoring ──────────────────────────────────────────────────────────────────
def summarize(results):
    n = len(results)
    if n == 0:
        return {"n": 0, "hit1_count": 0, "hit_at_1": 0.0, "hit1_ci": [0.0, 0.0], "hit_at_3": 0.0, "mrr": 0.0,
                "per_label": {}}
    hit1 = sum(1 for gold, ranked in results if ranked[:1] == [gold])
    hit3 = sum(1 for gold, ranked in results if gold in ranked[:3])
    rr = sum(1.0 / (ranked.index(gold) + 1) if gold in ranked else 0.0 for gold, ranked in results)
    lo, hi = wilson(hit1, n)
    per_label = {}
    for gold, ranked in results:
        stats = per_label.setdefault(gold, [0, 0])
        stats[1] += 1
        stats[0] += int(ranked[:1] == [gold])
    return {
        "n": n, "hit1_count": hit1,
        "hit_at_1": round(hit1 / n, 3), "hit1_ci": [round(lo, 3), round(hi, 3)],
        "hit_at_3": round(hit3 / n, 3), "mrr": round(rr / n, 3),
        "per_label": {k: {"hits": v[0], "total": v[1]} for k, v in per_label.items()},
    }


def print_coverage(docs):
    parsed = [d["parsed"] for d in docs]
    n = len(docs)
    print(f"\nParser coverage on {n} real logs:")
    print(f"   frameworks: {dict(Counter(p.framework for p in parsed))}")
    print(f"   error type found:  {sum(p.error_type != 'UnknownError' for p in parsed)}/{n}")
    print(f"   message found:     {sum(p.error_message != 'No message extracted' for p in parsed)}/{n}")
    print(f"   test name found:   {sum(p.test_name != 'UnknownTest' for p in parsed)}/{n}")
    print(f"   module found:      {sum(p.module != 'Unknown' for p in parsed)}/{n}")
    bad = [d for d in docs if d["parsed"].error_type == "UnknownError"]
    if bad:
        print("   logs with no error type (first 12):")
        for d in bad[:12]:
            print(safe(f"      {d['id']}: {d['key_line'][:100]}"))


def main():
    if not os.path.exists(CSV_PATH):
        print("No labels.csv. Finish Day 7 first.")
        return
    docs = load_documents()
    queries = [i for i, d in enumerate(docs) if d["label"] != "other"]
    if len(queries) < 10:
        print(f"Only {len(queries)} usable labeled logs; label more before evaluating.")
        return

    print(f"{len(docs)} verified logs in the index pool, {len(queries)} usable as queries "
          f"({len(docs) - len(queries)} are 'other' and act as distractors).")
    counts = Counter(docs[i]["label"] for i in queries)
    for label, count in counts.most_common():
        print(f"   {label:<24}{count:>3}{'   <-- thin' if count < 5 else ''}")
    print_coverage(docs)

    texts = {v: [d["parsed"].to_search_text(v) for d in docs] for v in VARIANTS}
    E = {v: embed_batch(texts[v]) for v in VARIANTS}
    tokens = [tokenize(t) for t in texts["no_names"]]

    systems = {f"embedding / {v}": embedding_system(E[v], docs) for v in VARIANTS}
    systems["bm25 / no_names"] = bm25_system(tokens, docs)
    systems["hybrid / no_names"] = hybrid_system(E["no_names"], tokens, docs)
    systems["majority class (baseline)"] = majority_system(docs)

    results = {}
    for name, fn in systems.items():
        outcome = [(docs[q]["label"], fn(q, allowed_indices(docs, q))) for q in queries]
        results[name] = summarize(outcome)

    print(f"\n{'system':<30}{'Hit@1 (95% CI)':>22}{'Hit@3':>8}{'MRR':>7}")
    for name, r in results.items():
        ci = f"{r['hit_at_1']:.0%} ({r['hit1_ci'][0]:.0%}-{r['hit1_ci'][1]:.0%})"
        print(f"{name:<30}{ci:>22}{r['hit_at_3']:>8.0%}{r['mrr']:>7.2f}")

    variant_names = [f"embedding / {v}" for v in VARIANTS]
    best = max(variant_names, key=lambda n: results[n]["hit1_count"])
    gain = results[best]["hit1_count"] - results["embedding / full"]["hit1_count"]
    if best != "embedding / full" and gain >= 3:
        print(f"\n>>> '{best.split(' / ')[1]}' beats 'full' by {gain} queries. Consider setting "
              f"DEFAULT_VARIANT = \"{best.split(' / ')[1]}\" in ingestion/log_parser.py (then delete data/faiss_store).")
    else:
        print("\n>>> No text variant clearly beats 'full' (differences under 3 queries are noise). Keep the default.")

    print(f"\nPer-class Hit@1 for {best}:")
    for label, s in results[best]["per_label"].items():
        print(f"   {label:<24}{s['hits']}/{s['total']}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "documents": len(docs), "queries": len(queries),
            "label_counts": dict(counts), "systems": results,
        }, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()