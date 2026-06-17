"""
Full experiment run → Result1/

Runs all 108 cells (C1, A1, A2, A3, A4) with max_tasks_per_cell=1 (smoke run).
Outputs:
  Result1/full_run.txt        — captured console log
  Result1/metrics_analysis.txt
  Result1/frugality_analysis.txt
  Result1/run_log.jsonl       — structured per-run records
  Result1/RESULTS.md          — complete narrative report
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
RESULT_DIR = ROOT / "Result1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
os.environ.setdefault("SWARM_ROLE", "researcher")

# ── Tee stdout/stderr into full_run.txt ──────────────────────────────────────

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()
    def fileno(self):
        return self._streams[0].fileno()

_log_file = open(RESULT_DIR / "full_run.txt", "w", encoding="utf-8", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

# ── Imports ───────────────────────────────────────────────────────────────────

from frugal_swarm.experiments.configurations import ALL_CONFIGS, enumerate_cells
from frugal_swarm.experiments.runner import run_cell, _SELF_ORG_CONFIGS
from frugal_swarm.experiments.metrics import compute_h2, compute_h3, hypothesis_table
from frugal_swarm.coordination.role_monitor import RoleMonitor
from frugal_swarm.memory.chroma_store import ChromaStore
from frugal_swarm.memory.mlflow_tracker import MLflowTracker
from frugal_swarm.model.ollama_client import OllamaClient
from config import ROLE_STABILITY_WINDOW, H1_ROLE_STABILITY_THRESHOLD, MODEL_POOL

# ── Run ───────────────────────────────────────────────────────────────────────

def _agent_ids(n):
    return [f"agent_{i}" for i in range(n)]


def main():
    t_start = time.perf_counter()
    print(f"\n{'='*70}")
    print(f"  Frugal AI Swarm — Result1 Experiment Run")
    print(f"  Date  : {date.today().isoformat()}")
    print(f"  Models: {MODEL_POOL}")
    print(f"  Cells : {len(enumerate_cells())} total (max_tasks_per_cell=1)")
    print(f"{'='*70}\n")

    client  = OllamaClient()
    chroma  = ChromaStore()
    mlflow_tracker = MLflowTracker()

    cells = enumerate_cells(ALL_CONFIGS)
    all_metrics = []
    run_records  = []   # structured per-run for run_log.jsonl
    errors       = []

    shared_role_monitors = {}
    for cfg in ALL_CONFIGS:
        if cfg.name in _SELF_ORG_CONFIGS:
            shared_role_monitors[cfg.name] = RoleMonitor(
                _agent_ids(cfg.swarm_size), window_size=ROLE_STABILITY_WINDOW
            )

    for idx, cell in enumerate(cells, 1):
        cfg    = cell["config"]
        family = cell["family"]
        seed   = cell["seed"]
        mode   = cell["verification_mode"]

        print(f"\n[{idx:>3}/{len(cells)}] {cfg.name} | {family} | seed={seed} | mode={mode.value}")
        t0 = time.perf_counter()
        try:
            m = run_cell(
                config=cfg,
                family=family,
                seed=seed,
                mode=mode,
                client=client,
                chroma=chroma,
                mlflow_tracker=mlflow_tracker,
                max_tasks=1,
                role_mon=shared_role_monitors.get(cfg.name),
                output_jsonl=str(RESULT_DIR / "task_outputs.jsonl"),
            )
            all_metrics.append(m)
            elapsed = time.perf_counter() - t0
            print(f"    => success | tasks={m.n_tasks} ok={m.n_success} "
                  f"tokens={m.total_tokens} elapsed={elapsed:.1f}s")
            run_records.append({
                "idx": idx, "config": cfg.name, "family": family,
                "seed": seed, "mode": mode.value,
                "n_tasks": m.n_tasks, "n_success": m.n_success,
                "total_tokens": m.total_tokens,
                "energy_kwh": m.energy_kwh_est,
                "queue_wait_s": m.queue_wait_s,
                "peak_ram_mb": m.peak_ram_mb,
                "ttfuo_s": m.ttfuo_s,
                "wall_s": elapsed,
                "status": "ok",
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            msg = str(exc)
            print(f"    => ERROR: {msg}")
            errors.append({"idx": idx, "config": cfg.name, "family": family,
                           "seed": seed, "mode": mode.value, "error": msg})
            run_records.append({
                "idx": idx, "config": cfg.name, "family": family,
                "seed": seed, "mode": mode.value, "status": "error", "error": msg,
                "wall_s": elapsed,
            })

    # ── Post-hoc H1 stability update for A3/A4 ───────────────────────────────
    for cfg_name, role_mon in shared_role_monitors.items():
        summary = role_mon.summary()
        for m in all_metrics:
            if m.config_name == cfg_name:
                m.mean_role_stability = summary.get("mean_role_stability", 0.0)
                m.specialisation_entropy = summary.get("specialisation_entropy", 0.0)
                m.h1_passed = None  # type: ignore[assignment]

    # ── H2 ────────────────────────────────────────────────────────────────────
    swarm_by_cell = defaultdict(dict)
    for m in all_metrics:
        if m.config_name in _SELF_ORG_CONFIGS:
            swarm_by_cell[(m.config_name, m.family, m.seed)][m.verification_mode] = m

    h2_computed = 0
    for key, mode_runs in swarm_by_cell.items():
        none_r = mode_runs.get("none")
        full_r = mode_runs.get("full")
        sel_r  = mode_runs.get("selective")
        if none_r and full_r and sel_r:
            h2_res = compute_h2(none_r, full_r, sel_r)
            for k, v in h2_res.items():
                setattr(sel_r, k, v)
            h2_computed += 1

    print(f"\n  H2 computed for {h2_computed} (config, family, seed) triplets")

    # ── H3 ────────────────────────────────────────────────────────────────────
    c1_by_cell = {}
    for m in all_metrics:
        if m.config_name == "C1" and m.verification_mode == "none":
            c1_by_cell[(m.family, m.seed)] = m

    h3_computed = 0
    for m in all_metrics:
        if m.config_name in _SELF_ORG_CONFIGS and m.verification_mode == "none":
            c1 = c1_by_cell.get((m.family, m.seed))
            if c1:
                h3_res = compute_h3(m, c1)
                for k, v in h3_res.items():
                    setattr(m, k, v)
                h3_computed += 1

    print(f"  H3 computed for {h3_computed} A3/A4-vs-C1 pairs\n")

    # ── Hypothesis table ──────────────────────────────────────────────────────
    table = hypothesis_table(all_metrics)
    print("\n" + "="*70)
    print("  Hypothesis Verification Table")
    print("="*70)
    for k, v in table.items():
        print(f"  {k}: {v}")

    # ── Write structured run log ──────────────────────────────────────────────
    jsonl_path = RESULT_DIR / "run_log.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in run_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nWritten: {jsonl_path}")

    # ── Generate text analysis files ──────────────────────────────────────────
    _write_metrics_analysis(run_records, RESULT_DIR)
    _write_frugality_analysis(run_records, RESULT_DIR)
    _write_results_md(run_records, all_metrics, table, h2_computed, h3_computed,
                      errors, RESULT_DIR)

    wall_total = time.perf_counter() - t_start
    print(f"\nTotal wall time: {wall_total/60:.1f} min")
    print(f"Results in: {RESULT_DIR}")

    _log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


# ── Analysis writers ──────────────────────────────────────────────────────────

def _mean(vals):
    return sum(vals) / len(vals) if vals else None

def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "n/a"


def _write_metrics_analysis(records, out_dir: Path):
    ok_recs = [r for r in records if r.get("status") == "ok"]
    by_cfg = defaultdict(list)
    for r in ok_recs:
        by_cfg[r["config"]].append(r)

    lines = [
        "Metrics Analysis by Configuration",
        "=" * 50,
        f"Generated: {date.today().isoformat()}",
        f"Total runs: {len(records)}  (ok={len(ok_recs)}, error={len(records)-len(ok_recs)})",
        "",
    ]
    for cfg in ["C1", "A1", "A2", "A3", "A4"]:
        group = by_cfg.get(cfg, [])
        if not group:
            lines.append(f"[{cfg}] — no data"); lines.append(""); continue
        tokens = [r["total_tokens"] for r in group if r.get("total_tokens") is not None]
        walls  = [r["wall_s"] for r in group if r.get("wall_s") is not None]
        n_ok   = sum(r.get("n_success", 0) for r in group)
        n_tot  = sum(r.get("n_tasks", 0) for r in group)

        lines.append(f"[{cfg}]  cells={len(group)}")
        lines.append(f"  tasks total / success : {n_tot} / {n_ok}")
        if tokens:
            lines.append(f"  tokens  mean={_mean(tokens):.1f}  total={sum(tokens)}")
        if walls:
            lines.append(f"  wall_s  mean={_mean(walls):.1f}s  total={sum(walls):.1f}s")

        # Per verification mode (A3/A4)
        by_mode = defaultdict(list)
        for r in group:
            by_mode[r["mode"]].append(r)
        if len(by_mode) > 1:
            for mode, mrs in sorted(by_mode.items()):
                mt = [r["total_tokens"] for r in mrs if r.get("total_tokens") is not None]
                lines.append(f"  [{mode}] n={len(mrs)}  mean_tokens={_fmt(_mean(mt), '.1f')}")
        lines.append("")

    path = out_dir / "metrics_analysis.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {path}")


def _write_frugality_analysis(records, out_dir: Path):
    ok_recs = [r for r in records if r.get("status") == "ok"]
    by_cfg  = defaultdict(list)
    for r in ok_recs:
        by_cfg[r["config"]].append(r)

    lines = [
        "Frugality Analysis — H4 Coordination Overhead",
        "=" * 50,
        f"Generated: {date.today().isoformat()}",
        "",
    ]

    c1_e = _mean([r["energy_kwh"] for r in by_cfg.get("C1", []) if r.get("energy_kwh") is not None])
    lines.append(f"C1 baseline mean energy : {_fmt(c1_e, '.6f')} kWh")
    lines.append("")
    lines.append(f"{'Config':<6} {'n':>4}  {'Energy(kWh)':>12}  {'Overhead':>10}  "
                 f"{'Queue(s)':>9}  {'RAM(MB)':>8}  {'TTFUO(s)':>9}  {'Wall(s)':>8}")
    lines.append("-" * 80)

    for cfg in ["C1", "A1", "A2", "A3", "A4"]:
        group = by_cfg.get(cfg, [])
        energies = [r["energy_kwh"]   for r in group if r.get("energy_kwh")   is not None]
        queues   = [r["queue_wait_s"] for r in group if r.get("queue_wait_s") is not None]
        rams     = [r["peak_ram_mb"]  for r in group if r.get("peak_ram_mb")  is not None]
        ttfuos   = [r["ttfuo_s"]      for r in group if r.get("ttfuo_s")      is not None]
        walls    = [r["wall_s"]       for r in group if r.get("wall_s")       is not None]
        me = _mean(energies)
        overhead_str = f"{me/c1_e:.2f}x" if (me and c1_e) else "n/a"
        lines.append(
            f"{cfg:<6} {len(group):>4}  {_fmt(me, '.6f'):>12}  {overhead_str:>10}  "
            f"{_fmt(_mean(queues), '.2f'):>9}  {_fmt(_mean(rams), '.1f'):>8}  "
            f"{_fmt(_mean(ttfuos), '.2f'):>9}  {_fmt(_mean(walls), '.1f'):>8}"
        )

    lines += ["", "Notes:",
              "  Energy is estimated from wall-clock time × TDP (hardware-specific).",
              "  Queue wait only appears when multiple tasks are batched in one cell.",
              "  TTFUO = Time-To-First-Useful-Output (first task completed).",
              "  Overhead = mean energy / C1 baseline."]

    path = out_dir / "frugality_analysis.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {path}")


def _write_results_md(records, all_metrics, hyp_table, h2_computed, h3_computed,
                      errors, out_dir: Path):
    ok_recs = [r for r in records if r.get("status") == "ok"]
    by_cfg  = defaultdict(list)
    for r in ok_recs:
        by_cfg[r["config"]].append(r)

    c1_e = _mean([r["energy_kwh"] for r in by_cfg.get("C1", []) if r.get("energy_kwh") is not None])
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Per-config frugality rows
    frug_rows = []
    for cfg in ["C1", "A1", "A2", "A3", "A4"]:
        group = by_cfg.get(cfg, [])
        energies = [r["energy_kwh"]   for r in group if r.get("energy_kwh")   is not None]
        queues   = [r["queue_wait_s"] for r in group if r.get("queue_wait_s") is not None]
        rams     = [r["peak_ram_mb"]  for r in group if r.get("peak_ram_mb")  is not None]
        ttfuos   = [r["ttfuo_s"]      for r in group if r.get("ttfuo_s")      is not None]
        walls    = [r["wall_s"]       for r in group if r.get("wall_s")       is not None]
        me = _mean(energies)
        overhead_str = f"{me/c1_e:.2f}×" if (me and c1_e) else "n/a"
        frug_rows.append(
            f"| {cfg} | {len(group)} | {_fmt(me, '.6f')} | {overhead_str} | "
            f"{_fmt(_mean(queues), '.2f')} | {_fmt(_mean(rams), '.1f')} | "
            f"{_fmt(_mean(ttfuos), '.2f')} | {_fmt(_mean(walls), '.1f')} |"
        )

    # Per-config token rows
    token_rows = []
    for cfg in ["C1", "A1", "A2", "A3", "A4"]:
        group = by_cfg.get(cfg, [])
        tokens = [r["total_tokens"] for r in group if r.get("total_tokens") is not None]
        n_ok   = sum(r.get("n_success", 0) for r in group)
        n_tot  = sum(r.get("n_tasks", 0)   for r in group)
        token_rows.append(
            f"| {cfg} | {len(group)} | {n_tot} | {n_ok} | "
            f"{_fmt(_mean(tokens), '.0f')} | {_fmt(sum(tokens) if tokens else None, '.0f')} |"
        )

    # Hypothesis rows
    hyp_rows = []
    for k, v in hyp_table.items():
        hyp_rows.append(f"| `{k}` | {v} |")

    # Error section
    error_section = ""
    if errors:
        error_section = "\n## Errors\n\n" + "\n".join(
            f"- Cell {e['idx']} ({e['config']}/{e['family']}/seed={e['seed']}/{e['mode']}): `{e['error']}`"
            for e in errors
        ) + "\n"

    # Audit trail status
    audit_path = ROOT / "data" / "audit" / "audit_trail.jsonl"
    if audit_path.exists():
        audit_lines = sum(1 for _ in audit_path.open(encoding="utf-8"))
        audit_status = f"Active — {audit_lines} records written to `data/audit/audit_trail.jsonl`"
    else:
        audit_status = "File not yet created (no runs completed)"

    md = f"""# Frugal AI Swarm — Result1 Experiment Report

**Run timestamp:** {run_ts}
**Run type:** Full smoke run (1 task per cell)
**Total cells:** {len(records)} planned / {len(ok_recs)} completed / {len(errors)} errors
**LLM backend:** Ollama (local, CPU)
**Hypothesis framework:** H1 – H4

---

## 1. Architecture Taxonomy

| Config | Name | Roles | Models | Verification |
|--------|------|-------|--------|-------------|
| C1  | Single-Agent Baseline | None | Shared | None only |
| A1  | Fixed-Role, Single Model | Aligner → Drafter → Verifier | Shared | None only |
| A2  | Fixed-Role, Multi-Model | Aligner → Drafter → Verifier | Per-role (qwen/gemma/phi3) | None only |
| A3  | Self-Organising, Single Model | Shapley + DAG, N=3 | Shared | None / Selective / Full |
| A4  | Self-Organising, Multi-Model | Shapley + DAG, N=3 | Per-agent (qwen/gemma/phi3) | None / Selective / Full |

**Cell counts:** C1=12, A1=12, A2=12, A3=36, A4=36 → **108 total**

All configurations run exclusively on **education task families**:
- `formative_assessment_drafting`
- `curriculum_question_generation`
- `lesson_adaptation`
- `knowledge_base_retrieval`

---

## 2. Education & Governance

| Property | Status |
|----------|--------|
| `requires_teacher_review` tag | ✅ Set on all cells (education families only) |
| RBAC enforced | ✅ `SWARM_ROLE=researcher` via env var |
| PII scrubber | ✅ Active on all audit entries |
| Audit trail | ✅ {audit_status} |
| Data retention policy | ✅ 90-day audit / 30-day ChromaDB |
| Low-watermark support | ✅ `get_run_params()` wired — throttles to 256 tok / 1 round on x86 institutional |

---

## 3. Run Summary

| Config | Cells Run | Tasks Issued | Tasks OK | Mean Tokens/Cell | Total Tokens |
|--------|-----------|--------------|----------|-----------------|-------------|
{chr(10).join(token_rows)}

---

## 4. Frugality Metrics (H4)

| Config | n | Mean Energy (kWh) | vs C1 | Mean Queue Wait (s) | Mean Peak RAM (MB) | Mean TTFUO (s) | Mean Wall (s) |
|--------|---|------------------|-------|--------------------|--------------------|---------------|--------------|
{chr(10).join(frug_rows)}

**H4 goal:** swarm coordination overhead ≤ 3× C1 wall-clock time.
Energy is estimated from wall-clock × hardware TDP (frugality.py).

---

## 5. Hypothesis Results

| Metric | Value |
|--------|-------|
{chr(10).join(hyp_rows) if hyp_rows else "| (no hypothesis metrics computed — check MLflow) | — |"}

### Hypothesis Definitions

| # | Hypothesis | Scope | What It Measures |
|---|-----------|-------|-----------------|
| H1 | Fixed-role stability > {H1_ROLE_STABILITY_THRESHOLD} | A1, A2 | Roles don't drift between rounds (trivially 1.0 for fixed-role) |
| H2 | Verification selective mode reduces token use vs full | A3, A4 | selective ≤ full tokens while retaining reliability |
| H3 | Swarm token efficiency ≥ 1.2× vs C1 baseline | A3, A4 vs C1 | output quality per token gained by coordination |
| H4 | Coordination wall-clock overhead ≤ 3× C1 | A1–A4 vs C1 | cost of multi-agent coordination in latency |

> **H1 note:** A3/A4 have no fixed roles — H1 is **N/A** for self-organising configs.
> Role stability is still measured as an exploratory metric via Shapley + RoleMonitor.

**H2 triplets computed:** {h2_computed}
**H3 A3/A4-vs-C1 pairs:** {h3_computed}

---

## 6. Model Assignment

| Agent Slot | A2 Role | A4 Agent | Model |
|-----------|---------|----------|-------|
| 0 | Curriculum Aligner | agent_0 | `qwen2.5:3b` |
| 1 | Fact-Checker & Drafter | agent_1 | `gemma2:2b` |
| 2 | Pedagogical Verifier | agent_2 | `phi3:mini` |

A1 and A3 use the shared Ollama default model for all agents.

---

## 7. Low-Watermark Mode

`get_run_params()` selects parameters at runtime based on hardware detection:

| Mode | max_tokens | max_rounds | ollama_timeout | serial_rounds |
|------|-----------|-----------|---------------|--------------|
| Standard (default) | 512 | 2 | 120s | False |
| Low-watermark (x86 institutional) | 256 | 1 | 60s | True |

These values are now threaded from `run_cell()` through to the graph builders,
so resource-constrained machines automatically throttle without code changes.

---

## 8. Files in This Result Set

| File | Description |
|------|-------------|
| `full_run.txt` | Raw console log (all runs, errors, hypothesis table) |
| `run_log.jsonl` | Structured per-run records (JSON Lines) |
| `metrics_analysis.txt` | Per-config token / latency summary |
| `frugality_analysis.txt` | Per-config energy / queue / RAM / TTFUO table |
| `RESULTS.md` | This file |

MLflow experiment data: `mlflow_runs/` (open with `mlflow ui`)
Audit trail: `data/audit/audit_trail.jsonl`
{error_section}
---

*Generated by run_result1.py — Frugal AI Agent Swarms, COL Phase-1 Pilot*
"""

    path = out_dir / "RESULTS.md"
    path.write_text(md, encoding="utf-8")
    print(f"Written: {path}")


if __name__ == "__main__":
    main()
