"""
Audit trail — append-only JSONL log of every swarm action.

Every agent invocation, verification step, and final output is logged with:
  - timestamp (ISO 8601 UTC)
  - session_id
  - action_type
  - actor (agent_id or "system")
  - task_id
  - scrubbed prompt/response text
  - token counts
  - config_name

The audit file is append-only. Entries are never modified or deleted by the
system (retention policy in retention.py controls archival / deletion schedule).

Storage: DATA_DIR/audit/audit_trail.jsonl (one JSON object per line)
"""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_DIR
from frugal_swarm.governance.pii_scrubber import scrub_dict

AUDIT_DIR = DATA_DIR / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_FILE = AUDIT_DIR / "audit_trail.jsonl"

_lock = threading.Lock()  # thread-safe appends


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_action(
    session_id: str,
    action_type: str,
    actor: str,
    task_id: str,
    payload: dict[str, Any],
    config_name: str = "unknown",
) -> None:
    """
    Append one audit record to the trail.

    payload is PII-scrubbed before writing.
    """
    record = {
        "timestamp":   _now_iso(),
        "session_id":  session_id,
        "action_type": action_type,
        "actor":       actor,
        "task_id":     task_id,
        "config":      config_name,
        "payload":     scrub_dict(payload),
    }
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with AUDIT_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_trail(limit: int = 100, session_id: str | None = None) -> list[dict]:
    """
    Read the most recent *limit* audit records, optionally filtered by session.
    Returns records in reverse chronological order (newest first).
    """
    if not AUDIT_FILE.exists():
        return []
    with AUDIT_FILE.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    records = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if session_id and rec.get("session_id") != session_id:
            continue
        records.append(rec)
        if len(records) >= limit:
            break
    return records


ACTION_TYPES = [
    "task_received",
    "agent_invoked",
    "agent_responded",
    "verification_triggered",
    "verification_result",
    "final_output",
    "run_complete",
    "pii_scrub_applied",
    "retention_applied",
]
