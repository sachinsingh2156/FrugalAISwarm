"""
Frugality metrics — H4 hypothesis: measure the *cost* of coordination.

Collected per swarm run and logged to MLflow alongside quality scores.

Metrics
-------
wall_clock_s     : total wall-clock time from task_started to run_complete
ttfuo_s          : Time-To-First-Useful-Output — seconds until first non-trivial
                   agent response arrives (useful for latency-sensitive deployments)
peak_ram_mb      : peak RSS memory during the run (requires psutil)
energy_kwh_est   : rough energy estimate via TDP proxy (CPU-only; no GPU assumed)
queue_wait_s     : cumulative time agents spent waiting for upstream context
                   (proxy for coordination overhead)

H4 null hypothesis: swarm coordination does NOT add disproportionate resource
cost relative to C1 (single-agent) baseline within the LOW_WATERMARK hardware tier.

Usage
-----
    collector = FrugalityCollector()
    collector.start()
    # ... run swarm ...
    collector.record_first_output()
    # ... finish ...
    metrics = collector.finish()
    print(metrics)  # FrugalitySnapshot
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


@dataclass
class FrugalitySnapshot:
    """Immutable record of frugality metrics for one swarm run."""
    wall_clock_s: float
    ttfuo_s: Optional[float]          # None if first output never recorded
    peak_ram_mb: Optional[float]      # None if psutil unavailable
    energy_kwh_est: Optional[float]   # None if no TDP data
    queue_wait_s: float               # cumulative coordination overhead

    def as_dict(self) -> dict:
        return {
            "wall_clock_s":    round(self.wall_clock_s, 3),
            "ttfuo_s":         round(self.ttfuo_s, 3) if self.ttfuo_s is not None else None,
            "peak_ram_mb":     round(self.peak_ram_mb, 1) if self.peak_ram_mb is not None else None,
            "energy_kwh_est":  round(self.energy_kwh_est, 6) if self.energy_kwh_est is not None else None,
            "queue_wait_s":    round(self.queue_wait_s, 3),
        }

    def __str__(self) -> str:
        d = self.as_dict()
        parts = [f"wall={d['wall_clock_s']}s"]
        if d["ttfuo_s"] is not None:
            parts.append(f"ttfuo={d['ttfuo_s']}s")
        if d["peak_ram_mb"] is not None:
            parts.append(f"peak_ram={d['peak_ram_mb']}MB")
        if d["energy_kwh_est"] is not None:
            parts.append(f"energy={d['energy_kwh_est']}kWh")
        parts.append(f"queue_wait={d['queue_wait_s']}s")
        return "FrugalitySnapshot(" + ", ".join(parts) + ")"


class FrugalityCollector:
    """
    Context-manager / manual collector for frugality metrics.

    Usage as context manager:
        with FrugalityCollector() as fc:
            fc.record_first_output()
            fc.add_queue_wait(1.2)
        snap = fc.snapshot

    Usage manually:
        fc = FrugalityCollector()
        fc.start()
        ...
        fc.record_first_output()
        fc.add_queue_wait(0.5)
        snap = fc.finish()
    """

    # Conservative TDP estimate for a typical x86 institutional server (W)
    # Source: dual-socket Xeon ~150 W per socket → 300 W total, no GPU
    DEFAULT_TDP_WATTS: float = 150.0

    def __init__(self, tdp_watts: float | None = None):
        self._tdp = tdp_watts or self.DEFAULT_TDP_WATTS
        self._t0: float | None = None
        self._t_first_output: float | None = None
        self._queue_wait_s: float = 0.0
        self._peak_ram_mb: float | None = None
        self._ram_monitor: _RamMonitor | None = None
        self.snapshot: FrugalitySnapshot | None = None

    def start(self) -> "FrugalityCollector":
        self._t0 = time.perf_counter()
        if _PSUTIL_AVAILABLE:
            self._ram_monitor = _RamMonitor()
            self._ram_monitor.start()
        return self

    def record_first_output(self) -> None:
        """Call this when the first meaningful agent response arrives."""
        if self._t_first_output is None and self._t0 is not None:
            self._t_first_output = time.perf_counter() - self._t0

    def add_queue_wait(self, seconds: float) -> None:
        """Accumulate time an agent spent waiting for upstream context."""
        self._queue_wait_s += seconds

    def finish(self) -> FrugalitySnapshot:
        assert self._t0 is not None, "FrugalityCollector.start() was never called"
        wall = time.perf_counter() - self._t0
        ttfuo = self._t_first_output

        if self._ram_monitor is not None:
            self._ram_monitor.stop()
            self._peak_ram_mb = self._ram_monitor.peak_mb

        # Energy estimate: TDP × wall-clock time → convert W·s → kWh
        energy_kwh = (self._tdp * wall) / 3_600_000.0

        self.snapshot = FrugalitySnapshot(
            wall_clock_s=wall,
            ttfuo_s=ttfuo,
            peak_ram_mb=self._peak_ram_mb,
            energy_kwh_est=energy_kwh,
            queue_wait_s=self._queue_wait_s,
        )
        return self.snapshot

    # ── Context manager ───────────────────────────────────────────────────────
    def __enter__(self) -> "FrugalityCollector":
        return self.start()

    def __exit__(self, *_) -> None:
        self.finish()


class _RamMonitor(threading.Thread):
    """Background thread that polls RSS every 100 ms and records the peak."""

    POLL_INTERVAL = 0.1  # seconds

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mb: float = 0.0
        self._stop_event = threading.Event()

    def run(self) -> None:
        proc = psutil.Process()
        while not self._stop_event.is_set():
            try:
                rss = proc.memory_info().rss / (1024 * 1024)  # bytes → MB
                if rss > self.peak_mb:
                    self.peak_mb = rss
            except psutil.NoSuchProcess:
                break
            self._stop_event.wait(self.POLL_INTERVAL)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=1.0)


# ── MLflow integration helper ─────────────────────────────────────────────────

def log_frugality_to_mlflow(snap: FrugalitySnapshot, prefix: str = "frugality") -> None:
    """Log a FrugalitySnapshot to the active MLflow run (no-op if unavailable)."""
    try:
        import mlflow
        for k, v in snap.as_dict().items():
            if v is not None:
                mlflow.log_metric(f"{prefix}/{k}", v)
    except Exception:
        pass  # MLflow not available or no active run — silently skip


# ── H4 threshold check ────────────────────────────────────────────────────────

def check_h4(snap: FrugalitySnapshot, baseline_wall_s: float) -> dict:
    """
    Compare a swarm run's frugality metrics against a C1 baseline.

    Returns a dict with pass/fail for H4 sub-checks:
      - wall_overhead_ratio: swarm_wall / baseline_wall  (threshold from config)
      - queue_fraction:      queue_wait / wall_clock     (should be < 0.20)
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    try:
        from config import H4_WALL_OVERHEAD_THRESHOLD
    except ImportError:
        H4_WALL_OVERHEAD_THRESHOLD = 3.0  # fallback: swarm OK if < 3× C1 wall time

    ratio = snap.wall_clock_s / max(baseline_wall_s, 0.001)
    queue_frac = snap.queue_wait_s / max(snap.wall_clock_s, 0.001)

    return {
        "wall_overhead_ratio": round(ratio, 3),
        "wall_overhead_pass":  ratio <= H4_WALL_OVERHEAD_THRESHOLD,
        "queue_fraction":      round(queue_frac, 3),
        "queue_fraction_pass": queue_frac < 0.20,
        "h4_pass":             ratio <= H4_WALL_OVERHEAD_THRESHOLD and queue_frac < 0.20,
    }
