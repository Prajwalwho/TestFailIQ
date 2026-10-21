
import os
import sys
import pickle
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from storage.faiss_store import (
    FAISSStore,
    INDEX_PATH,
    META_PATH,
)
from ingestion.log_parser import parse_log


def main():
    index_path = Path(INDEX_PATH)
    meta_path = Path(META_PATH)

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Existing FAISS index or metadata is missing. "
            "Nothing was changed."
        )

    # Load existing records without modifying the active store.
    with open(meta_path, "rb") as f:
        records = pickle.load(f)

    if not records:
        raise RuntimeError("No existing records found. Aborting.")

    print(f"Found {len(records)} existing records.")

    # Build a new index separately.
    rebuilt = FAISSStore()
    rebuilt.records = []

    for old_record in records:
        parsed = parse_log(
            old_record.log_text,
            test_name=old_record.test_name,
            module=old_record.module,
        )

        # Preserve the original record, including ID and feedback.
        rebuilt.add(old_record, parsed.to_search_text())

    if rebuilt.size != len(records):
        raise RuntimeError(
            f"Count mismatch: {rebuilt.size} vectors, "
            f"{len(records)} records."
        )

    # Back up the active files before replacing them.
    backup_dir = index_path.parent / "rebuild_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(index_path, backup_dir / index_path.name)
    shutil.copy2(meta_path, backup_dir / meta_path.name)

    rebuilt.save()

    print(f"Rebuilt index with {rebuilt.size} records.")
    print(f"Backup saved in: {backup_dir}")
    print("Existing record IDs and feedback were preserved.")


if __name__ == "__main__":
    main()
