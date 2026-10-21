"""
Collect real failed-CI logs from public GitHub Actions runs and build a labeling sheet.

PowerShell, from the backend folder (token once per terminal):
    $env:GITHUB_TOKEN = "<classic token with public_repo scope>"
    python data/collect_real_logs.py --per-repo 6 --target 90   # collect (resumable)
    python data/collect_real_logs.py --autolabel                # rule-based first pass, marked "auto" (unverified)
    python data/collect_real_logs.py --label                    # you label every row that has no label yet
    python data/collect_real_logs.py --audit                    # you verify a random sample of auto labels
    python data/collect_real_logs.py --summary                  # gold / silver counts and auto-label precision
    python data/collect_real_logs.py --samples                  # write data/real_samples.txt for parser work

Label states in labels.csv ("reviewed" column):
    yes   gold: verified by you
    auto  silver: assigned by keyword rules, not verified
    (empty) no label yet
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter

import httpx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(BASE_DIR, "real_logs")
LOG_DIR = os.path.join(REAL_DIR, "logs")
MANIFEST_PATH = os.path.join(REAL_DIR, "manifest.json")
CSV_PATH = os.path.join(REAL_DIR, "labels.csv")
SAMPLES_PATH = os.path.join(BASE_DIR, "real_samples.txt")
API = "https://api.github.com"

DEFAULT_REPOS = [
    "pytest-dev/pytest", "pandas-dev/pandas", "django/django", "scikit-learn/scikit-learn",
    "numpy/numpy", "home-assistant/core", "apache/airflow", "pallets/flask",
    "vitejs/vite", "vuejs/core", "expressjs/express",
    "spring-projects/spring-boot", "junit-team/junit5", "quarkusio/quarkus",
    "gin-gonic/gin", "grafana/grafana",
]

CATEGORIES = [
    "assertion_mismatch", "missing_dependency", "timeout", "network_or_service",
    "file_or_path", "null_or_type_error", "flaky_race", "resource_exhaustion",
    "build_or_compile", "config_or_env", "other",
]

# First matching rule becomes the suggested / auto category
SUGGEST_RULES = [
    ("missing_dependency", r"ModuleNotFoundError|ImportError|Cannot find module|No matching distribution|Could not resolve dependenc|cannot find package|NoClassDefFoundError|conflicting versions"),
    ("timeout", r"TimeoutError|timed out|Timeout of \d+|deadline exceeded|exceeded the \d+ ?(?:ms|s|minute)"),
    ("network_or_service", r"ConnectionRefused|Connection refused|ConnectionResetError|ECONNREFUSED|ECONNRESET|ETIMEDOUT|getaddrinfo|Name or service not known|HTTP 5\d\d|Bad Gateway|Service Unavailable|SSLError|Max retries exceeded|unexpected EOF|failed to fetch"),
    ("file_or_path", r"FileNotFoundError|ENOENT|No such file or directory|NoSuchFileException|FileNotFoundException"),
    ("resource_exhaustion", r"MemoryError|OutOfMemoryError|heap out of memory|No space left on device|OOMKilled"),
    ("build_or_compile", r"SyntaxError|COMPILATION ERROR|compilation failed|error TS\d+|cannot find symbol|error\[E\d+\]|Failed to execute goal .*compile|compile(?:Test)?Java FAILED"),
    ("null_or_type_error", r"NoneType|AttributeError|TypeError|NullPointerException|Cannot read propert|undefined is not|is not a function|nil pointer"),
    ("config_or_env", r"environment variable|Permission denied|401 Unauthorized|403 Forbidden|Authentication failed|version mismatch|requires Python|Unsupported engine"),
    ("assertion_mismatch", r"AssertionError|AssertionFailedError|Expected:|assert .* ==|toBe\(|toEqual\(|^--- FAIL:"),
]
SUGGEST_COMPILED = [(c, re.compile(p, re.I | re.M)) for c, p in SUGGEST_RULES]

# Logs containing these are too noisy for rule-based labeling; a human must decide
AMBIGUOUS = re.compile(r"INTERNALERROR|PR_BODY:|GITHUB_TOKEN:|Run Quality Checks on a PR", re.I)

# Jobs that are not about test failures
SKIP_JOBS = re.compile(
    r"lint|format|style|docs?\b|documentation|spell|license|pre-?commit|label|changelog|"
    r"codeql|dependabot|stale|benchmark|release|publish|deploy|"
    r"trivy|security|zizmor|scan|sarif|status|lock-?threads|triage|welcome|automerge|comment|"
    r"typing|type-?check|mypy|checkstyle|sonar", re.I)

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ?")

# A log is kept only if it contains a real test, compile or dependency failure signature
STRONG_SIGNAL = re.compile(
    r"^FAILED .+|^FAIL\s+\S+|^--- FAIL: |\bFAILED$|AssertionError|AssertionFailedError|"
    r"ERR_ASSERTION|Tests run:.*(?:Failures|Errors): [1-9]|\d+ (?:failing|failed)\b|● .+ ›|✕|"
    r"error TS\d+|COMPILATION ERROR|compile(?:Test)?Java FAILED|"
    r"ModuleNotFoundError|ImportError|Cannot find module|NoClassDefFoundError|"
    r"Traceback \(most recent|\w*(?:Exception|Error):", re.M)
NOISE_LINE = re.compile(
    r"^(INPUT_|##\[(?:group|endgroup|command|warning|debug)|env:|Cache |Received (?:\d+|string)|Expected substring)")
MARKERS = STRONG_SIGNAL

CSV_FIELDS = ["id", "category", "reviewed", "audit", "root_cause", "key_line",
              "suggested_category", "framework", "repo", "job_name", "run_url"]


# ── Log processing ───────────────────────────────────────────────────────────
def clean(text):
    return [TIMESTAMP.sub("", ANSI.sub("", line)).rstrip() for line in text.splitlines()]


def excerpt(lines, context=12, max_windows=6, max_lines=250):
    """Windows of lines around failure markers; failure detail is usually near the end."""
    windows = []
    for i, line in enumerate(lines):
        if MARKERS.search(line):
            lo, hi = max(0, i - context), min(len(lines), i + context + 1)
            if windows and lo <= windows[-1][1]:
                windows[-1][1] = max(windows[-1][1], hi)
            else:
                windows.append([lo, hi])
    out = []
    for lo, hi in windows[-max_windows:]:
        if out:
            out.append("   [...]")
        out.extend(lines[lo:hi])
    return out[-max_lines:]


def key_line_of(lines):
    for line in lines:
        s = line.strip()
        if not s or "Process completed" in s or NOISE_LINE.match(s):
            continue
        if STRONG_SIGNAL.search(s):
            return s[:200]
    return lines[0].strip()[:200] if lines else ""


def detect_framework(text):
    if re.search(r"> Task :\S+|BUILD FAILED|Gradle Test Run", text):
        return "gradle"
    if re.search(r"\[ERROR\] Failed to execute goal|Tests run:.*Failures", text):
        return "maven"
    if re.search(r"short test summary info|=+ FAILURES =+|pytest", text):
        return "pytest"
    if re.search(r"^--- FAIL|^FAIL\t|\bgo test\b", text, re.M):
        return "go"
    if re.search(r"Test Suites:|Test Files|jest|vitest", text, re.I):
        return "jest"
    if re.search(r"\bmocha\b|\d+ passing|\d+ failing", text, re.I):
        return "mocha"
    return "other"


def suggest(text):
    for category, pattern in SUGGEST_COMPILED:
        if pattern.search(text):
            return category
    return "other"


# ── GitHub API ───────────────────────────────────────────────────────────────
def gh_get(client, url, params=None):
    r = None
    for attempt in range(1, 6):
        try:
            r = client.get(url, params=params)
        except httpx.TransportError as e:
            wait = 5 * attempt
            print(f"  network error ({type(e).__name__}), retrying in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 403 and r.headers.get("x-ratelimit-remaining") == "0":
            wait = max(int(r.headers.get("x-ratelimit-reset", time.time() + 60)) - int(time.time()), 1) + 1
            print(f"  rate limited, sleeping {min(wait, 900)}s")
            time.sleep(min(wait, 900))
            continue
        time.sleep(0.3)
        return r
    if r is None:
        raise httpx.ConnectError("network unreachable after 5 attempts")
    return r


def collect_repo(client, repo, quota, known_jobs, key_counts, next_id, found):
    """Append new records to `found` (so a mid-repo crash does not lose them)."""
    r = gh_get(client, f"{API}/repos/{repo}/actions/runs", {"status": "failure", "per_page": 40})
    if r.status_code != 200:
        print(f"  {repo}: cannot list runs ({r.status_code}): {r.text[:120]}")
        return
    for run in r.json().get("workflow_runs", []):
        if len(found) >= quota:
            break
        jr = gh_get(client, f"{API}/repos/{repo}/actions/runs/{run['id']}/jobs",
                    {"filter": "latest", "per_page": 50})
        if jr.status_code != 200:
            continue
        failed = [j for j in jr.json().get("jobs", [])
                  if j.get("conclusion") == "failure" and not SKIP_JOBS.search(j["name"])]
        for job in failed[:2]:
            if (repo, job["id"]) in known_jobs or len(found) >= quota:
                continue
            lr = gh_get(client, f"{API}/repos/{repo}/actions/jobs/{job['id']}/logs")
            if lr.status_code != 200:
                continue
            lines = clean(lr.text)
            ex = excerpt(lines)
            if len(ex) < 5:
                continue
            key = key_line_of(ex)
            norm = re.sub(r"\d+", "N", key)
            if key_counts[norm] >= 2:  # avoid dozens of copies of one flaky failure
                continue
            key_counts[norm] += 1
            rec_id = f"REAL-{next_id + len(found):03d}"
            with open(os.path.join(LOG_DIR, f"{rec_id}.log"), "w", encoding="utf-8") as f:
                f.write("\n".join(ex))
            found.append({
                "id": rec_id, "repo": repo, "run_id": run["id"], "job_id": job["id"],
                "job_name": job["name"], "run_url": run["html_url"],
                "framework": detect_framework("\n".join(lines)), "key_line": key,
                "suggested_category": suggest("\n".join(ex)),
            })
            known_jobs.add((repo, job["id"]))


# ── Files ────────────────────────────────────────────────────────────────────
def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def read_rows():
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(rows):
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sync_csv(manifest):
    """Rewrite labels.csv from the manifest, keeping labels and review state already stored."""
    existing = {}
    if os.path.exists(CSV_PATH):
        existing = {row["id"]: row for row in read_rows()}
    rows = []
    for rec in manifest:
        old = existing.get(rec["id"], {})
        row = {k: rec.get(k, "") for k in CSV_FIELDS}
        for field in ("category", "reviewed", "audit", "root_cause"):
            row[field] = old.get(field, "") or ""
        rows.append(row)
    write_rows(rows)


def save_progress(manifest):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    sync_csv(manifest)


def safe(text):
    return text.encode("ascii", "replace").decode()


def load_excerpt(row):
    path = os.path.join(LOG_DIR, f"{row['id']}.log")
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    return path, lines


# ── Labeling ─────────────────────────────────────────────────────────────────
def auto_label():
    """Rule-based first pass. Marks rows 'auto' (silver). Ambiguous logs are left for you."""
    rows = read_rows()
    auto_count = left_count = 0
    for row in rows:
        if row.get("reviewed", "") in ("yes", "auto"):
            continue
        _, lines = load_excerpt(row)
        text = "\n".join(lines) if lines else row.get("key_line", "")
        category = suggest(text)
        if not AMBIGUOUS.search(text) and category != "other":
            row["category"] = category
            row["reviewed"] = "auto"
            auto_count += 1
        else:
            row["category"] = ""
            left_count += 1
    write_rows(rows)
    print(f"Auto-labeled {auto_count} rows (silver, unverified).")
    print(f"Left {left_count} rows with no label for you (--label).")


def show_row(row, position, total, lines):
    key = row["key_line"].strip()
    idx = next((i for i, l in enumerate(lines) if key and l.strip() == key), 0)
    print("=" * 100)
    print(safe(f"[{position}/{total}] {row['id']}  repo={row['repo']}  job={row['job_name']}"))
    print(f"current: {row['category'] or '(none)'}   suggested: {row['suggested_category']}")
    print("-" * 100)
    for line in lines[max(0, idx - 3): idx + 9]:
        print("   " + safe(line[:150]))


def ask_category(row, path, allow_enter):
    """Prompt until a category is chosen. Returns 'quit' or 'done'."""
    while True:
        choice = input("category> ").strip().lower()
        if choice == "q":
            return "quit"
        if choice == "o" and hasattr(os, "startfile"):
            if os.path.exists(path):
                os.startfile(path)
            continue
        if choice == "":
            if allow_enter and row["category"] in CATEGORIES + ["skip"]:
                return "done"
            print("  type a number, or s to skip")
            continue
        if choice == "s":
            row["category"] = "skip"
            return "done"
        if choice.isdigit() and 1 <= int(choice) <= len(CATEGORIES):
            row["category"] = CATEGORIES[int(choice) - 1]
            return "done"
        print("  not recognised")


def print_menu():
    menu = "  ".join(f"{i}={c}" for i, c in enumerate(CATEGORIES, 1))
    print(f"Categories: {menu}  s=skip (noise)")
    print("Enter=keep current (only when a label is shown)   o=open full log   q=save and quit\n")


def label_session(include_auto=False, ids=None):
    """Label rows that have no label yet (optionally also re-check auto rows, or specific ids)."""
    rows = read_rows()
    if ids:
        wanted = set(ids)
        todo = [r for r in rows if r["id"] in wanted]
    else:
        todo = [r for r in rows
                if r.get("reviewed", "") == "" or (include_auto and r.get("reviewed") == "auto")]
    if not todo:
        print("Nothing to label.")
        return
    print(f"{len(todo)} rows to label.")
    print_menu()
    for n, row in enumerate(todo, 1):
        path, lines = load_excerpt(row)
        show_row(row, n, len(todo), lines)
        result = ask_category(row, path, allow_enter=row.get("reviewed", "") in ("auto", "yes"))
        if result == "quit":
            write_rows(rows)
            print("Saved. Run --summary to see progress.")
            return
        row["reviewed"] = "yes"
        write_rows(rows)
    print("\nDone labeling.")


def audit_session(count=20):
    """Verify a random sample of auto labels; this measures how accurate the auto-labeler is."""
    rows = read_rows()
    pending = [r for r in rows if r.get("audit") == "pending"]
    if not pending:
        candidates = [r for r in rows if r.get("reviewed") == "auto" and not r.get("audit")]
        if not candidates:
            print("No auto-labeled rows left to audit.")
            return
        pending = random.Random(42).sample(candidates, min(count, len(candidates)))
        for r in pending:
            r["audit"] = "pending"
        write_rows(rows)
    print(f"{len(pending)} auto labels to verify. Press Enter if the shown label is right, otherwise pick the right one.")
    print_menu()
    for n, row in enumerate(pending, 1):
        path, lines = load_excerpt(row)
        before = row["category"]
        show_row(row, n, len(pending), lines)
        result = ask_category(row, path, allow_enter=True)
        if result == "quit":
            write_rows(rows)
            print("Saved. Run --audit again to continue.")
            return
        row["audit"] = "ok" if row["category"] == before else "fixed"
        row["reviewed"] = "yes"
        write_rows(rows)
    ok = sum(1 for r in rows if r.get("audit") == "ok")
    fixed = sum(1 for r in rows if r.get("audit") == "fixed")
    print(f"\nAudit complete: {ok} correct, {fixed} corrected -> auto-label precision {ok / (ok + fixed):.0%}")


def summary():
    if not os.path.exists(CSV_PATH):
        print("No labels.csv yet. Run the collector first.")
        return
    rows = read_rows()
    valid = set(CATEGORIES) | {"skip"}
    gold = [r for r in rows if r.get("reviewed") == "yes"]
    silver = [r for r in rows if r.get("reviewed") == "auto"]
    unlabeled = [r for r in rows if r.get("reviewed", "") == ""]
    ok = sum(1 for r in rows if r.get("audit") == "ok")
    fixed = sum(1 for r in rows if r.get("audit") == "fixed")
    pending = sum(1 for r in rows if r.get("audit") == "pending")

    print(f"Total {len(rows)} | gold (you verified) {len(gold)} | silver (auto, unverified) {len(silver)} "
          f"| no label yet {len(unlabeled)}")
    if ok + fixed:
        print(f"Auto-label audit: {ok} correct, {fixed} corrected -> precision {ok / (ok + fixed):.0%} "
              f"on {ok + fixed} audited rows")
    for r in gold + silver:
        if r["category"].strip() not in valid:
            print(f"Invalid category (fix): {r['id']}: '{r['category']}'")

    gold_counts = Counter(r["category"].strip() for r in gold)
    silver_counts = Counter(r["category"].strip() for r in silver)
    print(f"\n{'category':<22}{'gold':>5}{'silver':>8}")
    for c in CATEGORIES:
        flag = "   <-- few gold examples" if gold_counts[c] < 4 else ""
        print(f"{c:<22}{gold_counts[c]:>5}{silver_counts[c]:>8}{flag}")
    print(f"{'skip (noise)':<22}{gold_counts['skip']:>5}{silver_counts['skip']:>8}")

    others = [r for r in gold if r["category"].strip() == "other"]
    if others:
        print(f"\nKey lines of the {len(others)} gold 'other' logs (first 30):")
        for r in others[:30]:
            print(safe(f"   {r['id']}: {r['key_line'][:110]}"))

    if unlabeled:
        print(f"\nNext: python data/collect_real_logs.py --label     ({len(unlabeled)} rows have no label)")
    elif pending or (silver and ok + fixed == 0):
        print("\nNext: python data/collect_real_logs.py --audit")
    else:
        print("\nLabeling complete. Next: python data/collect_real_logs.py --samples")


def export_samples():
    """One log per repo (gold or silver, not skip), combined into data/real_samples.txt."""
    rows = read_rows()
    seen, parts = set(), []
    for row in rows:
        if row.get("category") in ("", "skip") or row["repo"] in seen:
            continue
        path, lines = load_excerpt(row)
        if not lines:
            continue
        parts.append(f"##### {row['id']} | repo={row['repo']} | job={row['job_name']} | "
                     f"category={row['category']} | framework={row['framework']}\n"
                     + "\n".join(lines[:120]) + "\n")
        seen.add(row["repo"])
    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Wrote {len(parts)} sample logs -> {SAMPLES_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--autolabel", action="store_true", help="rule-based first pass (silver labels)")
    parser.add_argument("--label", action="store_true", help="label rows that have no label yet")
    parser.add_argument("--include-auto", action="store_true", help="with --label: also re-check auto rows")
    parser.add_argument("--ids", nargs="*", help="with --label: re-open these specific ids, e.g. REAL-013")
    parser.add_argument("--audit", action="store_true", help="verify a random sample of auto labels")
    parser.add_argument("--count", type=int, default=20, help="audit sample size")
    parser.add_argument("--samples", action="store_true", help="write data/real_samples.txt")
    parser.add_argument("--target", type=int, default=100, help="stop collecting after this many logs")
    parser.add_argument("--per-repo", type=int, default=12)
    parser.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    args = parser.parse_args()

    if args.autolabel:
        auto_label()
        return
    if args.label:
        label_session(include_auto=args.include_auto, ids=args.ids)
        return
    if args.audit:
        audit_session(args.count)
        return
    if args.summary:
        summary()
        return
    if args.samples:
        export_samples()
        return

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print('Set a token first:  $env:GITHUB_TOKEN = "<token with public_repo scope>"')
        sys.exit(1)

    os.makedirs(LOG_DIR, exist_ok=True)
    manifest = load_manifest()
    known_jobs = {(m["repo"], m["job_id"]) for m in manifest}
    key_counts = Counter(re.sub(r"\d+", "N", m["key_line"]) for m in manifest)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        for repo in args.repos:
            if len(manifest) >= args.target:
                break
            already = sum(1 for m in manifest if m["repo"] == repo)
            if already >= args.per_repo:
                continue  # resuming: this repo is already done
            print(f"{repo} ...")
            new = []
            try:
                collect_repo(client, repo, args.per_repo - already, known_jobs, key_counts,
                             len(manifest) + 1, new)
            except httpx.HTTPError as e:
                print(f"  stopped early on {repo}: {type(e).__name__}: {e}")
            manifest.extend(new)
            save_progress(manifest)  # a crash now loses at most one repo
            print(f"  collected {len(new)} (total {len(manifest)})")

    save_progress(manifest)
    print(f"\nDone: {len(manifest)} logs.")
    print(f"Frameworks: {dict(Counter(m['framework'] for m in manifest))}")
    print("Next: python data/collect_real_logs.py --autolabel")


if __name__ == "__main__":
    main()