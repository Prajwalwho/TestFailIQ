"""
Rebuild the production FAISS index with the current parser and embedding variant.

Keeps every record's id, timestamp, resolution, known_issue_id and feedback. Refreshes the
parsed fields (error type, message, stack trace) and the vectors. test_name and module are
kept as stored (they may hold values a user typed in the Add to History form).
The old files are backed up first.

Stop the server before running this, then start it again.
"""

import pickle
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from ingestion.log_parser import parse_log
from storage.faiss_store import FAISSStore, INDEX_PATH, META_PATH


def main():
    index_path, meta_path = Path(INDEX_PATH), Path(META_PATH)
    if not index_path.exists() or not meta_path.exists():
        raise SystemExit("No existing index or metadata found. Start the server once to create it. Nothing changed.")

    with open(meta_path, "rb") as f:
        records = pickle.load(f)
    if not records:
        raise SystemExit("No records in the existing metadata. Nothing changed.")
    print(f"Found {len(records)} existing records.")

    rebuilt = FAISSStore()
    changed = 0
    for record in records:
        parsed = parse_log(record.log_text, test_name=record.test_name, module=record.module)
        if (record.error_type, record.error_message) != (parsed.error_type, parsed.error_message):
            changed += 1
        record.error_type = parsed.error_type
        record.error_message = parsed.error_message
        record.stacktrace = parsed.stacktrace
        rebuilt.add(record, parsed.to_search_text())

    if rebuilt.size != len(records):
        raise SystemExit(f"Count mismatch: {rebuilt.size} vectors for {len(records)} records. Nothing changed.")

    backup_dir = index_path.parent / "rebuild_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index_path, backup_dir / index_path.name)
    shutil.copy2(meta_path, backup_dir / meta_path.name)

    rebuilt.save()
    print(f"Rebuilt index with {rebuilt.size} records ({changed} had a different parsed error).")
    print(f"Backup saved in: {backup_dir}")


if __name__ == "__main__":
    main()