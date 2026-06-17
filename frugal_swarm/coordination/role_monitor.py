"""
Role-Stability and Anomaly Monitor.

Tracks per-agent task-type distributions across a sliding 100-task window.
Computes:
  - role_stability_score: fraction of windows in which an agent's modal task
    type stays constant  (H1 threshold: > 70 %)
  - specialisation_entropy: Shannon entropy of the task-type assignment
    matrix, normalised by log(N)  (goes to 0 when one agent monopolises,
    goes to 1 when uniform)
  - anomaly flags: agents whose embedding trajectory diverges sharply from
    their behavioural baseline (seed of the security-aware contribution filter)
"""
from __future__ import annotations

import math
from collections import Counter, deque
from typing import Any

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import ROLE_STABILITY_WINDOW
from frugal_swarm.model.embedder import cosine_similarity


class RoleMonitor:
    """
    Stateful monitor.  Call update() after each task completion.
    """

    def __init__(
        self,
        agent_ids: list[str],
        window_size: int = ROLE_STABILITY_WINDOW,
        anomaly_threshold: float = 0.30,  # cos-sim drop below baseline
    ) -> None:
        self.agent_ids = agent_ids
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold

        # agent_id → sliding window of task families (strings)
        self._family_window: dict[str, deque[str]] = {
            aid: deque(maxlen=window_size) for aid in agent_ids
        }
        # agent_id → sliding window of response embeddings
        self._embed_window: dict[str, deque[np.ndarray]] = {
            aid: deque(maxlen=window_size) for aid in agent_ids
        }
        # agent_id → modal family at end of each full window
        self._modal_history: dict[str, list[str]] = {aid: [] for aid in agent_ids}

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        agent_id: str,
        task_family: str,
        embedding: np.ndarray,
    ) -> None:
        """Record a completed task for agent_id."""
        self._family_window[agent_id].append(task_family)
        self._embed_window[agent_id].append(embedding)

        # Record modal family whenever the window is full
        win = self._family_window[agent_id]
        if len(win) == self.window_size:
            modal = Counter(win).most_common(1)[0][0]
            self._modal_history[agent_id].append(modal)

    # ── H1 metrics ────────────────────────────────────────────────────────────

    def role_stability_score(self) -> dict[str, float]:
        """
        Per-agent role-stability score: fraction of recorded full windows in
        which the modal task type is the same as the FIRST recorded modal type
        (i.e., the agent has "settled" into a specialisation).

        Returns {} if no windows have been completed yet.
        """
        scores: dict[str, float] = {}
        for aid in self.agent_ids:
            history = self._modal_history[aid]
            if not history:
                scores[aid] = 0.0
                continue
            reference = history[0]
            scores[aid] = sum(1 for m in history if m == reference) / len(history)
        return scores

    def mean_role_stability(self) -> float:
        """Mean role-stability score across all agents."""
        scores = list(self.role_stability_score().values())
        return float(np.mean(scores)) if scores else 0.0

    def specialisation_entropy(self) -> float:
        """
        Shannon entropy of the current task-type assignment matrix,
        normalised by log(N).

        Entropy → 0   when one agent monopolises all tasks of a type.
        Entropy → 1   when assignments are perfectly uniform across agents
                       and task types.
        """
        n_agents = len(self.agent_ids)
        if n_agents <= 1:
            return 0.0

        # Build (agent x task_type) count matrix
        all_families: set[str] = set()
        for aid in self.agent_ids:
            all_families.update(self._family_window[aid])
        if not all_families:
            return 0.0

        counts: dict[str, Counter] = {
            aid: Counter(self._family_window[aid]) for aid in self.agent_ids
        }

        # Compute per-agent distribution entropy then average
        total_entropy = 0.0
        for aid in self.agent_ids:
            total = sum(counts[aid].values())
            if total == 0:
                continue
            probs = [v / total for v in counts[aid].values()]
            h = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
            total_entropy += h

        max_entropy = math.log(len(all_families)) if len(all_families) > 1 else 1.0
        return float(total_entropy / (n_agents * max_entropy))

    # ── Anomaly detection ─────────────────────────────────────────────────────

    def anomaly_flags(self) -> dict[str, bool]:
        """
        Flag agents whose recent embedding diverges from their baseline.

        Baseline = mean of first half of the embedding window.
        Recent   = mean of the second half.
        Flagged if cosine similarity between baseline and recent < (1 - threshold).
        """
        flags: dict[str, bool] = {}
        for aid in self.agent_ids:
            window = list(self._embed_window[aid])
            if len(window) < 4:
                flags[aid] = False
                continue
            mid = len(window) // 2
            baseline_mean = np.mean(window[:mid], axis=0)
            recent_mean = np.mean(window[mid:], axis=0)
            # Normalise
            bn = np.linalg.norm(baseline_mean)
            rn = np.linalg.norm(recent_mean)
            if bn < 1e-9 or rn < 1e-9:
                flags[aid] = False
                continue
            sim = cosine_similarity(baseline_mean / bn, recent_mean / rn)
            flags[aid] = sim < (1.0 - self.anomaly_threshold)
        return flags

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        return {
            "role_stability_per_agent": self.role_stability_score(),
            "mean_role_stability": self.mean_role_stability(),
            "specialisation_entropy": self.specialisation_entropy(),
            "anomaly_flags": self.anomaly_flags(),
            "task_counts": {
                aid: dict(Counter(self._family_window[aid]))
                for aid in self.agent_ids
            },
        }
