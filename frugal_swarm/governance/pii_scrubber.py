"""
PII scrubber — pre-processing step for all prompts and logged responses.

Removes or redacts common PII patterns before any text is:
  - sent to an LLM
  - written to the audit trail
  - stored in ChromaDB / MLflow

Patterns covered:
  - Email addresses
  - UK/international phone numbers
  - UK National Insurance numbers
  - Student ID patterns (common HE formats)
  - Full name + DOB combos (heuristic)
  - IP addresses

All scrubbing is regex-based and conservative (false positives preferred over
leaking real PII). Scrubbed spans are replaced with typed placeholders so
downstream text remains coherent.

Note: This is a first-line defence, NOT a substitute for proper data governance.
Personal data should not enter the system in the first place.
"""
from __future__ import annotations

import re
from typing import NamedTuple


class ScrubResult(NamedTuple):
    text: str          # scrubbed text
    n_replacements: int  # how many spans were replaced


# ── Patterns ──────────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, str]] = [
    # Email
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
     "[EMAIL]"),
    # Phone (UK and international)
    (r"(?:\+44|0044|0)[\s\-]?(?:\d[\s\-]?){9,10}",
     "[PHONE]"),
    # UK National Insurance number
    (r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
     "[NI_NUMBER]"),
    # Student ID (common HE pattern: letter(s) + 6-9 digits)
    (r"\b[A-Z]{1,3}\d{6,9}\b",
     "[STUDENT_ID]"),
    # IPv4
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
     "[IP_ADDRESS]"),
    # UK postcode
    (r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
     "[POSTCODE]"),
    # Date of birth (common formats: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD)
    (r"\b(?:\d{2}[/\-]\d{2}[/\-]\d{4}|\d{4}[/\-]\d{2}[/\-]\d{2})\b",
     "[DOB]"),
]

_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _PATTERNS
]


def scrub(text: str) -> ScrubResult:
    """Apply all PII patterns to *text* and return a ScrubResult."""
    count = 0
    for pattern, replacement in _COMPILED:
        new_text, n = pattern.subn(replacement, text)
        text = new_text
        count += n
    return ScrubResult(text=text, n_replacements=count)


def scrub_text(text: str) -> str:
    """Convenience wrapper — return only the scrubbed string."""
    return scrub(text).text


def scrub_dict(d: dict) -> dict:
    """
    Recursively scrub string values in a dict.
    Non-string values are left unchanged.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = scrub_text(v)
        elif isinstance(v, dict):
            out[k] = scrub_dict(v)
        elif isinstance(v, list):
            out[k] = [scrub_text(i) if isinstance(i, str) else i for i in v]
        else:
            out[k] = v
    return out
