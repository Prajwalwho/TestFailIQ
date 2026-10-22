"""
RAG ablation: does retrieved evidence improve the LLM's diagnosis?

Conditions (same prompt, same log excerpt, same model, temperature 0, fixed seed):
  none : the evidence sections say "(none)"
  rag  : similar past failures + KB entry, selected exactly as the app does, from a leave-one-run-out
         index (a failure never retrieves its own CI run)

Automatic score: the model also names a failure category, compared with your hand label.
Human score: blind A/B ratings of diagnosis pairs (--rate).

Note: past failures carry a "hand-labeled category" hint (no written fixes exist for real logs), so this
mostly measures whether retrieved neighbours help the model classify the failure.

Requires Ollama running. Resumable: results are cached in data/ablation_runs.jsonl.

    python evaluation/run_rag_ablation.py --limit 10   # quick smoke test
    python evaluation/run_rag_ablation.py              # all labeled failures
    python evaluation/run_rag_ablation.py --rate       # blind A/B ratings
    python evaluation/run_rag_ablation.py --report     # print the results without calling the LLM
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import httpx
import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from evaluation.run_real_eval import allowed_indices, load_documents, safe
from evaluation.stats import sign_test_pvalue, wilson_interval
from ingestion.embedder import embed_batch
from rag import generator
from rag.retriever import KB_MIN_SCORE, KB_PROMPT_MIN_SCORE, SIMILAR_MIN_SCORE, RetrievedEvidence
from storage.faiss_store import FailureRecord
from storage.knowledge_base import get_kb

DATA_DIR = os.path.join(BACKEND_DIR, "data")
RUNS_PATH = os.path.join(DATA_DIR, "ablation_runs.jsonl")
RATINGS_PATH = os.path.join(DATA_DIR, "ablation_ratings.json")
RESULTS_PATH = os.path.join(DATA_DIR, "eval_rag_results.json")
RATING_TARGET = 20
EMPTY = RetrievedEvidence([], [], None, 0.0)


def hint(label):
    """Keep in sync with ingest_real_logs.py."""
    return f"hand-labeled category: {label}" if label and label != "other" else None


# ── Evidence (leave-one-run-out) ─────────────────────────────────────────────
def build_evidence(q, docs, E, kb):
    allowed = allowed_indices(docs, q)
    scores = E[allowed] @ E[q]
    similar = []
    for k in np.argsort(-scores, kind="stable")[:5]:
        score = float(scores[k])
        if score < SIMILAR_MIN_SCORE:
            break
        other = docs[int(allowed[k])]
        p = other["parsed"]
        similar.append((FailureRecord(
            id=other["id"], test_name=p.test_name, module=p.module, error_type=p.error_type,
            error_message=p.error_message, stacktrace=p.stacktrace, log_text=p.raw_log, timestamp="",
            known_issue_id=None, resolution=hint(other["label"])), score))
    matches = [(e, s) for e, s in kb.search(docs[q]["parsed"].to_search_text(), top_k=3) if s >= KB_MIN_SCORE]
    top_kb = matches[0][0] if matches and matches[0][1] >= KB_PROMPT_MIN_SCORE else None
    return RetrievedEvidence(similar, matches, top_kb, max((s for _, s in similar), default=0.0))


def has_evidence(evidence):
    return bool(evidence.similar_failures) or evidence.top_kb_entry is not None


# ── Cache ────────────────────────────────────────────────────────────────────
def load_cache():
    cache = {}
    if os.path.exists(RUNS_PATH):
        with open(RUNS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cache[(rec["id"], rec["condition"])] = rec
    return cache


def append_cache(rec):
    with open(RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def run_one(doc, evidence, condition):
    prompt = generator._build_prompt(doc["parsed"], evidence, ask_category=True)
    start = time.time()
    try:
        data, error = generator.call_llm(prompt, num_predict=600, temperature=0.0, seed=7), None
    except Exception as e:  # connection, timeout, invalid JSON
        data, error = None, f"{type(e).__name__}: {e}"
    return {"id": doc["id"], "condition": condition, "response": data, "error": error,
            "seconds": round(time.time() - start, 1), "has_evidence": has_evidence(evidence), "reused": False}


def ordered_queries(docs, limit):
    order = [i for i, d in enumerate(docs) if d["label"] != "other"]
    random.Random(5).shuffle(order)  # fixed order, so a bigger --limit extends a smaller run
    return order[:limit] if limit else order


# ── Scoring ──────────────────────────────────────────────────────────────────
def predicted_category(rec):
    data = rec.get("response") if rec else None
    if not data:
        return None
    return str(data.get("category") or "").strip().lower().replace(" ", "_").replace("-", "_")


def is_correct(rec, gold):
    return predicted_category(rec) == gold


def load_ratings():
    if os.path.exists(RATINGS_PATH):
        with open(RATINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def report(docs, queries, cache):
    rows = []
    for q in queries:
        d = docs[q]
        none, rag = cache.get((d["id"], "none")), cache.get((d["id"], "rag"))
        if none and rag:
            rows.append((d, none, rag))
    n = len(rows)
    if n == 0:
        print("No completed pairs yet. Run without --report first.")
        return
    gold = [d["label"] for d, _, _ in rows]
    none_ok = [is_correct(none, g) for (_, none, _), g in zip(rows, gold)]
    rag_ok = [is_correct(rag, g) for (_, _, rag), g in zip(rows, gold)]
    with_ev = [i for i, (_, _, rag) in enumerate(rows) if not rag.get("reused")]
    chance = max(Counter(gold).values()) / n

    def line(name, oks, recs):
        k = sum(oks)
        lo, hi = wilson_interval(k, len(oks))
        bad = sum(1 for r in recs if r["error"] or predicted_category(r) not in generator.CATEGORIES)
        secs = [r["seconds"] for r in recs if not r.get("reused")]
        avg = sum(secs) / len(secs) if secs else 0.0
        print(f"{name:<16}{k:>4}/{len(oks):<4}{k / len(oks):>8.0%} ({lo:.0%}-{hi:.0%}){bad:>9}{avg:>12.1f}")

    print(f"\n{n} failures scored. Chance level (largest class): {chance:.0%}. "
          f"Retrieval supplied evidence for {len(with_ev)} of them.")
    print(f"{'condition':<16}{'correct':>9}{'accuracy (95% CI)':>25}{'invalid':>9}{'avg sec':>12}")
    line("no retrieval", none_ok, [r for _, r, _ in rows])
    line("retrieval", rag_ok, [r for _, _, r in rows])

    b = sum(1 for a, c in zip(none_ok, rag_ok) if c and not a)
    c_ = sum(1 for a, c in zip(none_ok, rag_ok) if a and not c)
    print(f"\nPaired: retrieval right & plain wrong = {b}, plain right & retrieval wrong = {c_}, "
          f"sign test p = {sign_test_pvalue(b, c_):.3f}")
    if with_ev:
        print(f"Only the {len(with_ev)} failures where retrieval supplied evidence: "
              f"plain {sum(none_ok[i] for i in with_ev)}/{len(with_ev)}, "
              f"retrieval {sum(rag_ok[i] for i in with_ev)}/{len(with_ev)}")
    print("\nAccuracy per class (plain -> retrieval):")
    for label in sorted(set(gold)):
        idx = [i for i, g in enumerate(gold) if g == label]
        print(f"   {label:<24}{sum(none_ok[i] for i in idx)}/{len(idx)} -> {sum(rag_ok[i] for i in idx)}/{len(idx)}")

    ratings = load_ratings()
    summary = None
    if ratings:
        rag_w = sum(1 for r in ratings if r["choice"] == "rag")
        none_w = sum(1 for r in ratings if r["choice"] == "none")
        ties = sum(1 for r in ratings if r["choice"] == "tie")
        decisive = rag_w + none_w
        lo, hi = wilson_interval(rag_w, decisive)
        summary = {"retrieval_wins": rag_w, "plain_wins": none_w, "ties": ties,
                   "win_rate_among_decisive": round(rag_w / decisive, 3) if decisive else None,
                   "sign_test_p": round(sign_test_pvalue(rag_w, none_w), 3)}
        print(f"\nBlind A/B ratings ({len(ratings)}): retrieval better {rag_w}, plain better {none_w}, tie {ties}")
        if decisive:
            print(f"   retrieval wins {rag_w / decisive:.0%} of decisive comparisons "
                  f"(95% CI {lo:.0%}-{hi:.0%}, sign test p = {summary['sign_test_p']})")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(), "model": generator.OLLAMA_MODEL,
            "failures_scored": n, "with_evidence": len(with_ev), "chance_level": round(chance, 3),
            "plain_correct": sum(none_ok), "retrieval_correct": sum(rag_ok),
            "retrieval_right_plain_wrong": b, "plain_right_retrieval_wrong": c_,
            "sign_test_p": round(sign_test_pvalue(b, c_), 3), "ratings": summary,
        }, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")


# ── Blind rating ─────────────────────────────────────────────────────────────
def fmt(data):
    out = f"root cause: {data.get('root_cause')}\nexplanation: {data.get('explanation')}\n"
    for i, step in enumerate((data.get("next_steps") or [])[:3], 1):
        out += f"   {i}. {step}\n"
    if data.get("fix_snippet"):
        out += f"fix: {data['fix_snippet']}\n"
    return out


def rate_session(docs, queries, cache):
    ratings = load_ratings()
    done = {r["id"] for r in ratings}
    candidates = []
    for q in queries:
        d = docs[q]
        none, rag = cache.get((d["id"], "none")), cache.get((d["id"], "rag"))
        if (none and rag and not rag.get("reused") and none["response"] and rag["response"]
                and d["id"] not in done
                and none["response"]["root_cause"].strip() != rag["response"]["root_cause"].strip()):
            candidates.append(q)
    random.Random(11).shuffle(candidates)
    pool = candidates[: max(0, RATING_TARGET - len(ratings))]
    if not pool:
        print(f"Nothing to rate ({len(ratings)} ratings saved). Run the full evaluation first if it is empty.")
        return
    print(f"{len(pool)} comparisons to rate. For each, pick the diagnosis that would help an engineer fix THIS "
          "failure faster. Ignore style. If both are wrong or vague, call it a tie.")
    print("1 = A is better   2 = B is better   t = tie   s = skip   q = save and quit\n")
    for n, q in enumerate(pool, 1):
        d = docs[q]
        none, rag = cache[(d["id"], "none")], cache[(d["id"], "rag")]
        rag_is_a = random.Random(sum(map(ord, d["id"]))).random() < 0.5
        a, b = (rag, none) if rag_is_a else (none, rag)
        lines = d["parsed"].raw_log.splitlines()
        key = d["key_line"].strip()
        idx = next((i for i, l in enumerate(lines) if key and l.strip() == key), 0)
        print("=" * 100)
        print(safe(f"[{n}/{len(pool)}] {d['id']}"))
        for line in lines[max(0, idx - 2): idx + 8]:
            print("   " + safe(line[:140]))
        print("-" * 40 + " Diagnosis A " + "-" * 40)
        print(safe(fmt(a["response"])))
        print("-" * 40 + " Diagnosis B " + "-" * 40)
        print(safe(fmt(b["response"])))
        while True:
            choice = input("better> ").strip().lower()
            if choice in ("1", "2", "t", "s", "q"):
                break
            print("  type 1, 2, t, s or q")
        if choice == "q":
            break
        if choice == "s":
            continue
        verdict = "tie" if choice == "t" else ("rag" if (choice == "1") == rag_is_a else "none")
        ratings.append({"id": d["id"], "choice": verdict})
        with open(RATINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(ratings, f, indent=2)
    print(f"\n{len(ratings)} ratings saved. Run --report to see the summary.")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only the first N failures (fixed order)")
    parser.add_argument("--rate", action="store_true", help="blind A/B rating session")
    parser.add_argument("--report", action="store_true", help="print results from the cache")
    args = parser.parse_args()

    docs = load_documents()
    queries = ordered_queries(docs, args.limit)
    cache = load_cache()

    if args.report:
        report(docs, queries, cache)
        return
    if args.rate:
        rate_session(docs, ordered_queries(docs, 0), cache)
        return

    try:
        httpx.get("http://localhost:11434/api/tags", timeout=5)
    except httpx.HTTPError:
        raise SystemExit("Ollama is not reachable at localhost:11434. Start it first.")

    E = embed_batch([d["parsed"].to_search_text() for d in docs])
    kb = get_kb()
    print(f"{len(queries)} failures x 2 conditions with {generator.OLLAMA_MODEL}. Results are cached, "
          "so you can stop with Ctrl+C and resume.\n")
    for n, q in enumerate(queries, 1):
        d = docs[q]
        evidence = build_evidence(q, docs, E, kb)
        for condition, ev in (("none", EMPTY), ("rag", evidence)):
            if (d["id"], condition) in cache:
                continue
            if condition == "rag" and not has_evidence(evidence):
                rec = {**cache[(d["id"], "none")], "condition": "rag", "reused": True}  # identical prompt
            else:
                rec = run_one(d, ev, condition)
            append_cache(rec)
            cache[(d["id"], condition)] = rec
            status = "reused" if rec["reused"] else ("error" if rec["error"] else f"{rec['seconds']}s")
            print(f"[{n}/{len(queries)}] {d['id']} {condition:<5}{status}")
    report(docs, queries, cache)


if __name__ == "__main__":
    main()