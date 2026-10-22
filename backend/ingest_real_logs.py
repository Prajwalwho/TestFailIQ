"""
Add the hand-verified real failures to the production FAISS index so the app can retrieve them.

Stop the server first, run this from the backend folder, then start the server again.
Safe to re-run: ids already in the index are skipped.
"""

import csv
import os
import sys
from datetime import datetime, timezone

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from ingestion.log_parser import parse_log
from storage.faiss_store import FailureRecord, get_store

CSV_PATH = os.path.join(BACKEND_DIR, "data", "real_logs", "labels.csv")
LOG_DIR = os.path.join(BACKEND_DIR, "data", "real_logs", "logs")


def main():
    store = get_store()
    if store.size == 0:
        raise SystemExit("The index is empty. Start the server once so the synthetic logs are indexed, "
                         "stop it, then run this again.")

    existing = {r.id for r in store.get_all()}
    added = skipped = 0
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("reviewed") != "yes" or row["category"].strip() in ("", "skip"):
                continue
            if row["id"] in existing:
                skipped += 1
                continue
            path = os.path.join(LOG_DIR, f"{row['id']}.log")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as lf:
                text = lf.read()
            parsed = parse_log(text)
            record = FailureRecord(
                id=row["id"],
                test_name=parsed.test_name,
                module=parsed.module,
                error_type=parsed.error_type,
                error_message=parsed.error_message,
                stacktrace=parsed.stacktrace,
                log_text=text,
                timestamp=datetime.now(timezone.utc).isoformat(),
                known_issue_id=None,
                resolution=row.get("root_cause", "").strip() or None,
            )
            store.add(record, parsed.to_search_text())
            added += 1

    store.save()
    print(f"Added {added} real failures ({skipped} already present). Index now holds {store.size} records.")


if __name__ == "__main__":
    main()