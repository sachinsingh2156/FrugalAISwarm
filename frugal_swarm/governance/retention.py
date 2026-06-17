"""
Data retention policy for audit trails and swarm artefacts.

Policy (configurable via config.py):
  - Audit records older than AUDIT_RETENTION_DAYS are moved to cold archive.
  - Task prompts/responses in ChromaDB are purged after CHROMA_RETENTION_DAYS.
  - No student-identifiable data is retained beyond the session by default.

This module provides the enforcement functions; scheduling is handled separately
(e.g., cron or the schedule skill).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_DIR

AUDIT_DIR = DATA_DIR / "audit"
ARCHIVE_DIR = DATA_DIR / "audit" / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Default retention windows (override in config.py)
DEFAULT_AUDIT_RETENTION_DAYS = 90
DEFAULT_CHROMA_RETENTION_DAYS = 30


def _get_retention_days(key: str, default: int) -> int:
    try:
        from config import AUDIT_RETENTION_DAYS, CHROMA_RETENTION_DAYS
        return AUDIT_RETENTION_DAYS if key == "audit" else CHROMA_RETENTION_DAYS
    except ImportError:
        return default


def apply_audit_retention(dry_run: bool = False) -> dict:
    """
    Move audit records older than retention window to the archive folder.

    Returns a summary dict: {archived: N, retained: N, dry_run: bool}
    """
    retention_days = _get_retention_days("audit", DEFAULT_AUDIT_RETENTION_DAYS)
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    audit_file = AUDIT_DIR / "audit_trail.jsonl"

    if not audit_file.exists():
        return {"archived": 0, "retained": 0, "dry_run": dry_run}

    retain_lines = []
    archive_lines = []

    with audit_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ts = datetime.fromisoformat(rec["timestamp"])
                if ts < cutoff:
                    archive_lines.append(line)
                else:
                    retain_lines.append(line)
            except (json.JSONDecodeError, KeyError, ValueError):
                retain_lines.append(line)  # keep malformed lines

    if not dry_run and archive_lines:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_path = ARCHIVE_DIR / f"audit_{stamp}.jsonl"
        with archive_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(archive_lines) + "\n")
        with audit_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(retain_lines) + "\n")

    return {
        "archived": len(archive_lines),
        "retained": len(retain_lines),
        "dry_run":  dry_run,
        "retention_days": retention_days,
    }


if __name__ == "__main__":
    result = apply_audit_retention(dry_run="--dry-run" in sys.argv)
    print(json.dumps(result, indent=2))
