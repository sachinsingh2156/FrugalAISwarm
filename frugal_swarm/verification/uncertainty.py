"""
Uncertainty signal for selective verification.

The uncertainty signal is derived from the mean token log-probability of a
generation, normalised per task type using a brief calibration pass.

High uncertainty (score → 1) → verify
Low uncertainty  (score → 0) → pass through

If Ollama does not return logprobs (older builds), we fall back to output
length as a proxy (shorter responses = more uncertain).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import UNCERTAINTY_THRESHOLD


class UncertaintyNormaliser:
    """
    Per-task-type normaliser.

    On a calibration pass the caller feeds (family, uncertainty_raw) pairs.
    Thereafter score() returns a value in [0, 1] using per-family min/max
    normalisation.
    """

    def __init__(self) -> None:
        self._mins: dict[str, float] = defaultdict(lambda: math.inf)
        self._maxs: dict[str, float] = defaultdict(lambda: -math.inf)

    def calibrate(self, family: str, uncertainty_raw: float) -> None:
        self._mins[family] = min(self._mins[family], uncertainty_raw)
        self._maxs[family] = max(self._maxs[family], uncertainty_raw)

    def score(self, family: str, uncertainty_raw: float | None) -> float:
        """
        Return normalised uncertainty in [0, 1].
        If calibration data is missing or raw is None, return 0.5 (neutral).
        """
        if uncertainty_raw is None:
            return 0.5
        lo = self._mins.get(family, uncertainty_raw)
        hi = self._maxs.get(family, uncertainty_raw)
        if abs(hi - lo) < 1e-9:
            return 0.5
        return float(max(0.0, min(1.0, (uncertainty_raw - lo) / (hi - lo))))

    def should_verify(
        self,
        family: str,
        uncertainty_raw: float | None,
        threshold: float = UNCERTAINTY_THRESHOLD,
    ) -> bool:
        """Return True if the output should be verified."""
        return self.score(family, uncertainty_raw) >= threshold
