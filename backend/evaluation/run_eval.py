"""
Held-out retrieval evaluation: embeddings vs BM25 keywords vs hybrid.

Builds a TEMPORARY in-memory index, queries it with held-out logs, and scores the
results against labels (known_issue_id). Never touches the production index on disk.

Run from the backend folder:
    python evaluation/run_eval.py              # index = eval_set.json "index" split
    python evaluation/run_eval.py --index prod # index = the production synthetic logs
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from rank_bm25 import BM25Okapi
from ingestion.log_parser import parse_log
from storage.faiss_store import FAISSStore, FailureRecord
from storage.knowledge_base import get_kb
from rag.retriever import SIMILAR_MIN_SCORE, KB_MIN_SCORE

DATA_DIR = os.path.join(BACKEND_DIR, "data")
EVAL_SET_PATH = os.path.join(DATA_DIR, "eval_set.json")
HARD_SET_PATH = os.path.join(DATA_DIR, "eval_hard.json")
PROD_META_PATH = os.path.join(DATA_DIR, "synthetic_logs_metadata.json")
TOP_K = 5
RRF_K = 60
SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_index_entries(source):
    if source == "prod":
        return [
            {"id": e["id"], "known_issue_id": e["known_issue_id"], "log_text": e["log_text"]}
            for e in load_json(PROD_META_PATH)
        ]
    return [e for e in load_json(EVAL_SET_PATH) if e["split"] == "index"]


def make_queries(entries):
    queries = []
    for e in entries:
        p = parse_log(e["log_text"])
        queries.append({"id": e["id"], "gold": e["known_issue_id"], "text": p.to_search_text()})
    return queries


# ── Tokenizing for BM25 ──────────────────────────────────────────────────────
def tokenize(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)  # split camelCase
    return re.findall(r"[a-z0-9]+", text.lower())


# ── Retrieval systems: query -> full ranking [(item_id, score), ...] ─────────
def build_failure_side(index_entries):
    store = FAISSStore()  # fresh in-memory store; never saved
    ids, label_of, texts = [], {}, []
    for e in index_entries:
        p = parse_log(e["log_text"])
        record = FailureRecord(
            id=e["id"], test_name=p.test_name, module=p.module, error_type=p.error_type,
            error_message=p.error_message, stacktrace=p.stacktrace, log_text=e["log_text"],
            timestamp=now(), known_issue_id=e["known_issue_id"], resolution=None,
        )
        store.add(record, p.to_search_text())
        ids.append(e["id"])
        label_of[e["id"]] = e["known_issue_id"]
        texts.append(p.to_search_text())
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    n = len(ids)

    def emb(q):
        return [(rec.id, s) for rec, s in store.search(q, top_k=n)]

    def bm(q):
        scores = bm25.get_scores(tokenize(q))
        return sorted(zip(ids, (float(s) for s in scores)), key=lambda x: -x[1])

    return emb, bm, label_of, store.size


def build_kb_side():
    kb = get_kb()
    entries = kb.get_all()
    ids = [e.id for e in entries]
    bm25 = BM25Okapi([tokenize(e.to_search_text()) for e in entries])

    def emb(q):
        return [(en.id, s) for en, s in kb.search(q, top_k=len(entries))]

    def bm(q):
        scores = bm25.get_scores(tokenize(q))
        return sorted(zip(ids, (float(s) for s in scores)), key=lambda x: -x[1])

    return emb, bm, {i: i for i in ids}


def hybrid(emb, bm, k=RRF_K):
    """Rank by reciprocal rank fusion of both rankings; report the embedding cosine as the score."""
    def fused(q):
        emb_ranked = emb(q)
        cosine = dict(emb_ranked)
        fused_scores = {}
        for ranked in (emb_ranked, bm(q)):
            for rank, (item, _) in enumerate(ranked, 1):
                fused_scores[item] = fused_scores.get(item, 0.0) + 1.0 / (k + rank)
        order = sorted(fused_scores, key=lambda i: -fused_scores[i])
        return [(item, cosine[item]) for item in order]
    return fused


# ── Scoring ──────────────────────────────────────────────────────────────────
def to_issue_ranking(ranked, label_of):
    """Collapse record-level ranking to issue-level: first (best) occurrence of each label."""
    seen, out = set(), []
    for item, score in ranked:
        label = label_of[item]
        if label not in seen:
            seen.add(label)
            out.append((label, score))
    return out[:TOP_K]


def reciprocal_rank(ranked, gold):
    for i, (label, _) in enumerate(ranked, 1):
        if label == gold:
            return 1.0 / i
    return 0.0


def separation_auc(pos_scores, neg_scores):
    """P(a real match's top-1 score > a no-match log's top-1 score). 50% = coin flip."""
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos_scores) * len(neg_scores))


def at_threshold(rows, t):
    pos = [(q, r) for q, r in rows if q["gold"]]
    neg = [(q, r) for q, r in rows if not q["gold"]]
    correct = sum(1 for q, r in pos if r and r[0][1] >= t and r[0][0] == q["gold"])
    wrong = sum(1 for q, r in pos if r and r[0][1] >= t and r[0][0] != q["gold"])
    neg_fp = sum(1 for _, r in neg if r and r[0][1] >= t)
    return {
        "threshold": t,
        "recall": round(correct / len(pos), 3) if pos else 0.0,
        "wrong": round(wrong / len(pos), 3) if pos else 0.0,
        "negative_false_match": round(neg_fp / len(neg), 3) if neg else 0.0,
    }


def evaluate(rank_fn, label_of, queries, scale, threshold):
    rows = [(q, to_issue_ranking(rank_fn(q["text"]), label_of)) for q in queries]
    pos = [(q, r) for q, r in rows if q["gold"]]
    neg = [(q, r) for q, r in rows if not q["gold"]]

    hit1 = hit3 = 0
    rr_sum = 0.0
    misses = []
    for q, ranked in pos:
        labels = [label for label, _ in ranked]
        gold = q["gold"]
        ok1 = bool(labels) and labels[0] == gold
        hit1 += ok1
        hit3 += gold in labels[:3]
        rr_sum += reciprocal_rank(ranked, gold)
        if not ok1:
            misses.append({
                "id": q["id"], "gold": gold,
                "predicted": labels[0] if labels else None,
                "top1_score": round(ranked[0][1], 3) if ranked else 0.0,
                "gold_rank": labels.index(gold) + 1 if gold in labels else None,
            })

    n = len(pos)
    top1_pos = [r[0][1] for _, r in pos if r]
    top1_neg = [r[0][1] for _, r in neg if r]
    auc = separation_auc(top1_pos, top1_neg)
    result = {
        "scale": scale,
        "n_positive": n,
        "n_negative": len(neg),
        "hit_at_1": round(hit1 / n, 3) if n else 0.0,
        "hit_at_3": round(hit3 / n, 3) if n else 0.0,
        "mrr": round(rr_sum / n, 3) if n else 0.0,
        "separation_auc": round(auc, 3) if auc is not None else None,
        "misses": misses,
    }
    if scale == "cosine":
        sweep = [at_threshold(rows, t) for t in SWEEP]
        best = max(sweep, key=lambda s: (s["recall"] - s["wrong"] - s["negative_false_match"], -s["threshold"]))
        result.update({
            "current_threshold": threshold,
            "at_current_threshold": at_threshold(rows, threshold),
            "suggested_threshold": best["threshold"],
            "sweep": sweep,
        })
    return result


# ── Reporting ────────────────────────────────────────────────────────────────
def print_set(name, res):
    print(f"\n##### {name.upper()} ({res['n_queries']} logs) #####")
    print(f"{'system':<22}{'Hit@1':>7}{'Hit@3':>7}{'MRR':>7}{'AUC':>7}")
    for sys_name, r in res["systems"].items():
        auc = "-" if r["separation_auc"] is None else f"{r['separation_auc']:.0%}"
        print(f"{sys_name:<22}{r['hit_at_1']:>7.0%}{r['hit_at_3']:>7.0%}{r['mrr']:>7.2f}{auc:>7}")


def print_tuning(combined):
    for name in ("failures_embedding", "kb_embedding"):
        r = combined["systems"][name]
        cur = r["at_current_threshold"]
        print(f"\n--- {name}: threshold sweep on standard + hard ---")
        print(f"Current threshold {r['current_threshold']}: correct {cur['recall']:.0%}, "
              f"wrong {cur['wrong']:.0%}, no-match wrongly matched {cur['negative_false_match']:.0%}")
        print("threshold | correct | wrong | no-match false match")
        for s in r["sweep"]:
            print(f"   {s['threshold']:.2f}   |  {s['recall']:>4.0%}   | {s['wrong']:>4.0%}  | {s['negative_false_match']:>4.0%}")
        print(f"Suggested threshold: {r['suggested_threshold']}")
    f = combined["systems"]["failures_embedding"]["suggested_threshold"]
    k = combined["systems"]["kb_embedding"]["suggested_threshold"]
    print(f"\n>>> Suggested constants for rag/retriever.py:  SIMILAR_MIN_SCORE = {f}   KB_MIN_SCORE = {k}")


def print_misses(res):
    for name in ("failures_embedding", "kb_embedding"):
        misses = res["systems"][name]["misses"]
        if misses:
            print(f"\nHard-set top-1 misses for {name}:")
            for m in misses:
                print(f"   {m['id']}: expected {m['gold']}, got {m['predicted']} "
                      f"(score {m['top1_score']}, correct answer ranked #{m['gold_rank']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=["eval", "prod"], default="eval")
    args = parser.parse_args()

    index_entries = load_index_entries(args.index)
    standard = [e for e in load_json(EVAL_SET_PATH) if e["split"] == "test"]
    hard = load_json(HARD_SET_PATH)

    # Leak check: no test log may share its text with an index log
    index_texts = {e["log_text"] for e in index_entries}
    assert not index_texts & {e["log_text"] for e in standard + hard}, "test log also in index"
    print(f"Leak check passed. Index source: {args.index} ({len(index_entries)} logs).")

    f_emb, f_bm, f_labels, index_size = build_failure_side(index_entries)
    k_emb, k_bm, k_labels = build_kb_side()
    systems = {
        "failures_embedding": (f_emb, f_labels, "cosine", SIMILAR_MIN_SCORE),
        "failures_bm25": (f_bm, f_labels, "bm25", None),
        "failures_hybrid": (hybrid(f_emb, f_bm), f_labels, "cosine", SIMILAR_MIN_SCORE),
        "kb_embedding": (k_emb, k_labels, "cosine", KB_MIN_SCORE),
        "kb_bm25": (k_bm, k_labels, "bm25", None),
        "kb_hybrid": (hybrid(k_emb, k_bm), k_labels, "cosine", KB_MIN_SCORE),
    }

    query_sets = {
        "standard": make_queries(standard),
        "hard": make_queries(hard),
    }
    query_sets["combined"] = query_sets["standard"] + query_sets["hard"]

    results = {"generated_at": now(), "index_source": args.index, "index_size": index_size, "test_sets": {}}
    for set_name, queries in query_sets.items():
        results["test_sets"][set_name] = {
            "n_queries": len(queries),
            "systems": {
                name: evaluate(fn, labels, queries, scale, thr)
                for name, (fn, labels, scale, thr) in systems.items()
            },
        }

    for set_name in ("standard", "hard"):
        print_set(set_name, results["test_sets"][set_name])
    print_misses(results["test_sets"]["hard"])
    print_tuning(results["test_sets"]["combined"])

    out = "eval_results.json" if args.index == "eval" else "eval_results_prod.json"
    with open(os.path.join(DATA_DIR, out), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {os.path.join(DATA_DIR, out)}")


if __name__ == "__main__":
    main()