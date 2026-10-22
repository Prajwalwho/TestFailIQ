"""
Rebuild the production FAISS index in one step.

* keeps every existing record that is not a real log (the synthetic logs and anything added through
  the Add to History form) and re-embeds it with the current parser, preserving ids and feedback
* (re)adds all hand-verified real failures with their hand-labeled category as the resolution hint,
  followed by your root_cause note when you wrote one
* backs up the old index first

Stop the server first, run from the backend folder, then start the server again. Idempotent.
"""

import csv
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from evaluation.run_real_eval import COARSE
from ingestion.log_parser import parse_log
from storage.faiss_store import FAISSStore, FailureRecord, INDEX_PATH, META_PATH

CSV_PATH = BACKEND_DIR / "data" / "real_logs" / "labels.csv"
LOG_DIR = BACKEND_DIR / "data" / "real_logs" / "logs"


def hint(label, root_cause):
    """Keep in sync with evaluation/run_rag_ablation.py."""
    parts = []
    if label and label != "other":
        parts.append(f"hand-labeled category: {label}")
    if root_cause:
        parts.append(root_cause)
    return "; ".join(parts) or None


def main():
    index_path, meta_path = Path(INDEX_PATH), Path(META_PATH)
    if not index_path.exists() or not meta_path.exists():
        raise SystemExit("No index found. Start the server once so the synthetic logs are indexed, stop it, "
                         "then run this again.")
    with open(meta_path, "rb") as f:
        old = pickle.load(f)
    keep = [r for r in old if not r.id.startswith("REAL-")]
    if not keep:
        raise SystemExit("No non-real records found. Start the server once to index the synthetic logs first.")

    rebuilt = FAISSStore()
    for record in keep:
        parsed = parse_log(record.log_text, test_name=record.test_name, module=record.module)
        record.error_type = parsed.error_type
        record.error_message = parsed.error_message
        record.stacktrace = parsed.stacktrace
        rebuilt.add(record, parsed.to_search_text())

    added = 0
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            category = row["category"].strip()
            if row.get("reviewed") != "yes" or category in ("", "skip"):
                continue
            path = LOG_DIR / f"{row['id']}.log"
            if not path.exists():
                continue
            label = "other" if category == "other" else COARSE.get(category)
            text = path.read_text(encoding="utf-8")
            parsed = parse_log(text)
            rebuilt.add(
                FailureRecord(
                    id=row["id"],
                    test_name=parsed.test_name,
                    module=parsed.module,
                    error_type=parsed.error_type,
                    error_message=parsed.error_message,
                    stacktrace=parsed.stacktrace,
                    log_text=text,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    known_issue_id=None,
                    resolution=hint(label, row.get("root_cause", "").strip()),
                ),
                parsed.to_search_text(),
            )
            added += 1

    backup_dir = index_path.parent / "rebuild_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index_path, backup_dir / index_path.name)
    shutil.copy2(meta_path, backup_dir / meta_path.name)
    rebuilt.save()
    print(f"Kept {len(keep)} existing records, added {added} real failures. Index now holds {rebuilt.size} records.")
    print(f"Backup saved in: {backup_dir}")


if __name__ == "__main__":
    main()