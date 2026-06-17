"""
Corpus loader: reads task-family JSON files and returns Task objects.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import CORPUS_DIR
from frugal_swarm.coordination.state import Task

FAMILY_FILES = {
    # Original families
    "multistep_qa":                 CORPUS_DIR / "multistep_qa.json",
    "document_analysis":            CORPUS_DIR / "document_analysis.json",
    "workflow_planning":            CORPUS_DIR / "workflow_planning.json",
    # Education-domain families (supervisor recommendation: Ricky Cheng)
    "formative_assessment_drafting":  CORPUS_DIR / "formative_assessment_drafting.json",
    "curriculum_question_generation": CORPUS_DIR / "curriculum_question_generation.json",
    "lesson_adaptation":              CORPUS_DIR / "lesson_adaptation.json",
    "knowledge_base_retrieval":       CORPUS_DIR / "knowledge_base_retrieval.json",
}

EDUCATION_FAMILIES = [
    "formative_assessment_drafting",
    "curriculum_question_generation",
    "lesson_adaptation",
    "knowledge_base_retrieval",
]


def load_family(family: str, seed: int | None = None) -> list[Task]:
    """Load all tasks for one family."""
    path = FAMILY_FILES[family]
    raw: list[dict] = json.loads(path.read_text())
    if seed is not None:
        rng = random.Random(seed)
        raw = rng.sample(raw, k=len(raw))  # shuffle with seed
    return [
        Task(
            task_id=item["id"],
            family=item["family"],
            prompt=item["prompt"],
            reference=item["reference"],
            priority=len(raw) - i,  # first in list = highest priority
        )
        for i, item in enumerate(raw)
    ]


def load_corpus(
    families: list[str] | None = None,
    seed: int | None = None,
    max_per_family: int | None = None,
) -> list[Task]:
    """
    Load all tasks across the specified families (default: all families).
    Optionally limit to max_per_family items per family.
    """
    if families is None:
        families = list(FAMILY_FILES.keys())
    tasks: list[Task] = []
    for family in families:
        family_tasks = load_family(family, seed=seed)
        if max_per_family is not None:
            family_tasks = family_tasks[:max_per_family]
        tasks.extend(family_tasks)
    return tasks


def load_education_corpus(
    seed: int | None = None,
    max_per_family: int | None = None,
) -> list[Task]:
    """Convenience: load only the education-domain families."""
    return load_corpus(
        families=EDUCATION_FAMILIES,
        seed=seed,
        max_per_family=max_per_family,
    )
