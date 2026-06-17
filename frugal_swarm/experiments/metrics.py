"""
Metric computation for hypothesis verification.

All six metrics from Section 6.7.7 of the synopsis are computed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (
    H1_ROLE_STABILITY_THRESHOLD,
    H2_TOKEN_REDUCTION_THRESHOLD,
    H2_RELIABILITY_RETENTION,
    H3_TOKEN_EFFICIENCY_RATIO,
)
from frugal_swarm.coordination.state import Task
from frugal_swarm.corpus.rubrics import is_success, score


@dataclass
class RunMetrics:
    """Metrics for one experimental cell (config × family × seed × mode)."""
    run_id: str
    config_name: str
    family: str
    seed: int
    verification_mode: str
    swarm_size: int

    # Raw counts
    n_tasks: int = 0
    n_success: int = 0
    total_tokens: int = 0
    total_latency_s: float = 0.0
    n_verified: int = 0
    verification_tokens: int = 0

    # Derived
    task_success_rate: float = 0.0
    mean_tokens_per_task: float = 0.0
    mean_latency_per_task: float = 0.0
    token_efficiency_ratio: float = 0.0   # vs C1 baseline

    # H1
    mean_role_stability: float = 0.0
    specialisation_entropy: float = 0.0
    h1_passed: bool = False

    # H2 (computed across mode comparison)
    verification_token_overhead: float | None = None
    reliability_gain_retention: float | None = None
    h2_passed: bool | None = None

    # H3
    h3_passed: bool | None = None

    # H4 — frugality (energy / queueing overhead), wired in from FrugalityCollector
    energy_kwh_est: float | None = None
    queue_wait_s: float | None = None
    ttfuo_s: float | None = None
    peak_ram_mb: float | None = None

    # Hardware / deployment tier, wired in from hardware.get_hardware_snapshot()
    platform_tag: str = "unknown"
    low_watermark_mode: bool = False

    # Governance — RBAC enforcement actually exercised at submission time
    rbac_role: str = "researcher"
    rbac_granted: bool = True

    # Human-in-the-loop flag — True for education-domain families. Outputs from
    # these families are drafting aids only; no automated grading/assessment
    # decision is made without teacher review (per supervisor guidance, 5 Jun).
    requires_teacher_review: bool = False


def compute_run_metrics(
    run_id: str,
    config_name: str,
    family: str,
    seed: int,
    verification_mode: str,
    swarm_size: int,
    completed_tasks: list[Task],
    role_stability: float,
    specialisation_entropy: float,
) -> RunMetrics:
    """Compute per-run metrics from completed task objects."""
    m = RunMetrics(
        run_id=run_id,
        config_name=config_name,
        family=family,
        seed=seed,
        verification_mode=verification_mode,
        swarm_size=swarm_size,
    )
    m.n_tasks = len(completed_tasks)
    if m.n_tasks == 0:
        return m

    successes = [
        is_success(t.family, t.result or "", t.reference)
        for t in completed_tasks
    ]
    m.n_success = sum(successes)
    m.task_success_rate = m.n_success / m.n_tasks
    m.total_tokens = sum(t.tokens_used for t in completed_tasks)
    m.total_latency_s = sum(t.latency_s for t in completed_tasks)
    m.mean_tokens_per_task = m.total_tokens / m.n_tasks
    m.mean_latency_per_task = m.total_latency_s / m.n_tasks
    m.n_verified = sum(1 for t in completed_tasks if t.verified)

    m.mean_role_stability = role_stability
    m.specialisation_entropy = specialisation_entropy
    m.h1_passed = m.mean_role_stability > H1_ROLE_STABILITY_THRESHOLD

    return m


def compute_h2(
    none_run: RunMetrics,
    full_run: RunMetrics,
    selective_run: RunMetrics,
) -> dict[str, Any]:
    """
    Compute H2 quantities from three verification mode runs on the same
    (config, family, seed) cell.

    Returns a dict with h2 fields to merge into selective_run.
    """
    if none_run.mean_tokens_per_task == 0 or full_run.mean_tokens_per_task == 0:
        return {"verification_token_overhead": None, "reliability_gain_retention": None, "h2_passed": None}

    # Token reduction: (full_tokens - selective_tokens) / full_tokens
    token_reduction = (
        full_run.mean_tokens_per_task - selective_run.mean_tokens_per_task
    ) / full_run.mean_tokens_per_task

    # Reliability gain retention:
    # (selective_success - none_success) / (full_success - none_success)
    full_delta = full_run.task_success_rate - none_run.task_success_rate
    if abs(full_delta) < 1e-9:
        retention = 1.0  # no gain to preserve
    else:
        retention = (
            selective_run.task_success_rate - none_run.task_success_rate
        ) / full_delta

    h2_passed = (
        token_reduction >= H2_TOKEN_REDUCTION_THRESHOLD
        and retention >= H2_RELIABILITY_RETENTION
    )

    return {
        "verification_token_overhead": token_reduction,
        "reliability_gain_retention": retention,
        "h2_passed": h2_passed,
    }


def compute_h3(swarm_run: RunMetrics, c1_run: RunMetrics) -> dict[str, Any]:
    """
    Compute H3 token efficiency ratio.
    Ratio = (swarm_success × c1_tokens) / (c1_success × swarm_tokens)
    """
    if (
        c1_run.task_success_rate == 0
        or swarm_run.mean_tokens_per_task == 0
        or c1_run.mean_tokens_per_task == 0
    ):
        return {"token_efficiency_ratio": None, "h3_passed": None}

    ratio = (
        swarm_run.task_success_rate * c1_run.mean_tokens_per_task
    ) / (
        c1_run.task_success_rate * swarm_run.mean_tokens_per_task
    )

    return {
        "token_efficiency_ratio": ratio,
        "h3_passed": ratio >= H3_TOKEN_EFFICIENCY_RATIO,
    }


_FIXED_ROLE_CONFIGS = {"A1", "A2"}


def hypothesis_table(all_metrics: list[RunMetrics]) -> dict[str, Any]:
    """
    Aggregate all run metrics into the hypothesis-verification table.
    Returns a summary dict suitable for logging to MLflow or printing.
    """
    # H1 is meaningful only for fixed-role configs (A1, A2); trivially 1.0 there.
    h1_scores = [m.mean_role_stability for m in all_metrics if m.config_name in _FIXED_ROLE_CONFIGS]
    h1_mean = sum(h1_scores) / len(h1_scores) if h1_scores else 0.0
    h1_passed_rate = sum(1 for m in all_metrics if m.h1_passed and m.config_name in _FIXED_ROLE_CONFIGS) / max(len(h1_scores), 1)

    h2_results = [m for m in all_metrics if m.h2_passed is not None]
    h2_passed_rate = sum(1 for m in h2_results if m.h2_passed) / max(len(h2_results), 1)

    h3_results = [m for m in all_metrics if m.h3_passed is not None]
    h3_passed_rate = sum(1 for m in h3_results if m.h3_passed) / max(len(h3_results), 1)

    # H4: swarm wall-clock overhead vs C1 (energy proxy, since wall ∝ energy on fixed HW)
    from config import H4_WALL_OVERHEAD_THRESHOLD
    c1_energy = [m.energy_kwh_est for m in all_metrics
                 if m.config_name == "C1" and m.energy_kwh_est is not None]
    swarm_energy = [m.energy_kwh_est for m in all_metrics
                    if m.config_name != "C1" and m.energy_kwh_est is not None]
    c1_mean_e   = sum(c1_energy)   / len(c1_energy)   if c1_energy   else None
    swarm_mean_e = sum(swarm_energy) / len(swarm_energy) if swarm_energy else None
    if c1_mean_e and swarm_mean_e and c1_mean_e > 0:
        h4_overhead = swarm_mean_e / c1_mean_e
        h4_passed = h4_overhead <= H4_WALL_OVERHEAD_THRESHOLD
    else:
        h4_overhead = None
        h4_passed = None

    return {
        "H1_mean_role_stability": h1_mean,
        "H1_passed_fraction": h1_passed_rate,
        "H1_verdict": "SUPPORTED" if h1_mean > H1_ROLE_STABILITY_THRESHOLD else "NOT SUPPORTED",
        "H2_passed_fraction": h2_passed_rate,
        "H2_verdict": "SUPPORTED" if h2_passed_rate >= 0.5 else "NOT SUPPORTED",
        "H3_passed_fraction": h3_passed_rate,
        "H3_verdict": "SUPPORTED" if h3_passed_rate >= 0.5 else "NOT SUPPORTED",
        "H4_energy_overhead": round(h4_overhead, 2) if h4_overhead is not None else "n/a",
        "H4_verdict": ("SUPPORTED" if h4_passed else "NOT SUPPORTED") if h4_passed is not None else "n/a",
    }
