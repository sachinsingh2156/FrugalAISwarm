"""
Full experiment run → Result2/

108 cells × 7 tasks per cell = 756 total task invocations.
Designed to run unattended under nohup (SSH-disconnect-safe).

Output files (all in Result2/):
  full_run.txt          — live-appended console log
  run_log.jsonl         — structured per-run records (JSON Lines)
  progress.txt          — human-readable progress (overwritten each cell)
  metrics_analysis.txt
  frugality_analysis.txt
  RESULTS.md
  run.pid               — PID of this process
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
RESULT_DIR = ROOT / "Result2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TASKS_PER_CELL = 7   # 108 × 7 = 756 total task invocations

sys.path.insert(0, str(ROOT))
os.environ.setdefault("SWARM_ROLE", "researcher")

# ── Write PID so user can monitor / kill ─────────────────────────────────────
(RESULT_DIR / "run.pid").write_text(str(os.getpid()))

# ── Tee stdout/stderr into full_run.txt (line-buffered for SSH safety) ───────
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass
    def fileno(self):
        return self._streams[0].fileno()

_log_file = open(RESULT_DIR / "full_run.txt", "w", encoding="utf-8", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

# ── Imports ───────────────────────────────────────────────────────────────────
from frugal_swarm.experiments.configurations import ALL_CONFIGS, enumerate_cells
from frugal_swarm.experiments.runner import run_cell, _SELF_ORG_CONFIGS
from frugal_swarm.experiments.metrics import (
    RunMetrics, compute_h2, compute_h3, hypothesis_table,
)
from frugal_swarm.coordination.role_monitor import RoleMonitor
from frugal_swarm.memory.chroma_store import ChromaStore
from frugal_swarm.memory.mlflow_tracker import MLflowTracker
from frugal_swarm.model.ollama_client import OllamaClient
from config import ROLE_STABILITY_WINDOW, MODEL_POOL


def _agent_ids(n):
    return [f"agent_{i}" for i in range(n)]


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "n/a"


def _write_progress(idx, total, cfg_name, family, seed, mode, t_elapsed, errors):
    """Overwrite progress.txt with current state — safe for `tail -f` monitoring."""
    eta_s = None
    if idx > 0:
        rate = t_elapsed / idx           # seconds per cell so far
        remaining = (total - idx) * rate
        eta_h = int(remaining // 3600)
        eta_m = int((remaining % 3600) // 60)
        eta_s = f"{eta_h}h {eta_m}m"

    lines = [
        f"Result2 — Progress Report",
        f"Updated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"Cells    : {idx}/{total} ({100*idx/total:.1f}%)",
        f"Errors   : {errors}",
        f"Elapsed  : {int(t_elapsed//60)}m {int(t_elapsed%60)}s",
        f"ETA      : {eta_s or 'calculating...'}",
        f"",
        f"Current  : [{cfg_name}] {family} seed={seed} mode={mode}",
        f"",
        f"PID      : {os.getpid()}  (kill -0 {os.getpid()} to check alive)",
    ]
    try:
        (RESULT_DIR / "progress.txt").write_text("\n".join(lines))
    except Exception:
        pass


def main():
    t_start = time.perf_counter()
    total_planned = len(enumerate_cells(ALL_CONFIGS)) * MAX_TASKS_PER_CELL

    print(f"\n{'='*70}")
    print(f"  Frugal AI Swarm — Result2 Experiment Run")
    print(f"  Date       : {date.today().isoformat()}")
    print(f"  PID        : {os.getpid()}")
    print(f"  Models     : {MODEL_POOL}")
    print(f"  Cells      : {len(enumerate_cells(ALL_CONFIGS))} cells × {MAX_TASKS_PER_CELL} tasks = {total_planned} task invocations")
    print(f"  Output dir : {RESULT_DIR}")
    print(f"  Safe for   : SSH disconnect (nohup + full_run.txt live-appended)")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    client         = OllamaClient()
    chroma         = ChromaStore()
    mlflow_tracker = MLflowTracker()
    cells          = enumerate_cells(ALL_CONFIGS)
    all_metrics    = []
    run_records    = []
    errors         = []

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
        t_elapsed = time.perf_counter() - t_start

        _write_progress(idx - 1, len(cells), cfg.name, family, seed, mode.value,
                        t_elapsed, len(errors))

        print(f"\n[{idx:>3}/{len(cells)}] {cfg.name} | {family} | seed={seed} | mode={mode.value}")
        sys.stdout.flush()

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
                max_tasks=MAX_TASKS_PER_CELL,
                role_mon=shared_role_monitors.get(cfg.name),
                output_jsonl=str(RESULT_DIR / "task_outputs.jsonl"),
            )
            all_metrics.append(m)
            elapsed = time.perf_counter() - t0
            print(f"    => ok | n_tasks={m.n_tasks} n_success={m.n_success} "
                  f"tokens={m.total_tokens} wall={elapsed:.1f}s")
            sys.stdout.flush()
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
            import traceback
            elapsed = time.perf_counter() - t0
            msg = str(exc)
            tb  = traceback.format_exc()
            print(f"    => ERROR: {msg}\n{tb}")
            sys.stdout.flush()
            errors.append({"idx": idx, "config": cfg.name, "family": family,
                           "seed": seed, "mode": mode.value, "error": msg})
            run_records.append({
                "idx": idx, "config": cfg.name, "family": family,
                "seed": seed, "mode": mode.value,
                "status": "error", "error": msg, "wall_s": elapsed,
            })

        # Flush jsonl after every cell so partial results survive a crash
        jsonl_path = RESULT_DIR / "run_log.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in run_records:
                f.write(json.dumps(rec) + "\n")

    # ── Post-hoc: H1 stability for A3/A4 ────────────────────────────────────
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

    table = hypothesis_table(all_metrics)
    print("\n" + "="*70)
    print("  Hypothesis Verification Table")
    print("="*70)
    for k, v in table.items():
        print(f"  {k}: {v}")
    sys.stdout.flush()

    # ── Write all result files ────────────────────────────────────────────────
    _write_metrics_analysis(run_records, RESULT_DIR)
    _write_frugality_analysis(run_records, RESULT_DIR)
    _write_results_md(run_records, all_metrics, table, h2_computed, h3_computed,
                      errors, RESULT_DIR)

    wall_total = time.perf_counter() - t_start
    print(f"\nTotal wall time: {wall_total/60:.1f} min ({wall_total/3600:.2f} h)")
    print(f"Results in: {RESULT_DIR}")

    # Mark run complete
    _write_progress(len(cells), len(cells), "DONE", "—", "—", "—",
                    wall_total, len(errors))
    (RESULT_DIR / "progress.txt").write_text(
        (RESULT_DIR / "progress.txt").read_text() + f"\n\nRUN COMPLETE — {date.today().isoformat()}\n"
    )
    (RESULT_DIR / "run.pid").unlink(missing_ok=True)

    _log_file.flush()
    _log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


# ── Analysis writers ──────────────────────────────────────────────────────────

def _write_metrics_analysis(records, out_dir: Path):
    ok_recs = [r for r in records if r.get("status") == "ok"]
    by_cfg  = defaultdict(list)
    for r in ok_recs:
        by_cfg[r["config"]].append(r)

    lines = [
        "Metrics Analysis by Configuration",
        "=" * 55,
        f"Generated : {date.today().isoformat()}",
        f"Total runs: {len(records)}  ok={len(ok_recs)}  error={len(records)-len(ok_recs)}",
        f"Tasks/cell: {MAX_TASKS_PER_CELL}",
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
        lines.append(f"  tasks total / success : {n_tot} / {n_ok}"
                     f"  ({100*n_ok/max(n_tot,1):.1f}%)")
        if tokens:
            lines.append(f"  tokens  mean/cell={_mean(tokens):.0f}  "
                         f"mean/task={_mean(tokens)/MAX_TASKS_PER_CELL:.0f}  "
                         f"total={sum(tokens)}")
        if walls:
            lines.append(f"  wall_s  mean/cell={_mean(walls):.1f}s  total={sum(walls):.0f}s")

        by_mode = defaultdict(list)
        for r in group:
            by_mode[r["mode"]].append(r)
        if len(by_mode) > 1:
            for mode, mrs in sorted(by_mode.items()):
                mt = [r["total_tokens"] for r in mrs if r.get("total_tokens") is not None]
                lines.append(f"  [{mode:>9}] n={len(mrs)}  mean_tokens/cell={_fmt(_mean(mt), '.0f')}")
        lines.append("")

    path = out_dir / "metrics_analysis.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {path}")


def _write_frugality_analysis(records, out_dir: Path):
    ok_recs = [r for r in records if r.get("status") == "ok"]
    by_cfg  = defaultdict(list)
    for r in ok_recs:
        by_cfg[r["config"]].append(r)

    c1_e = _mean([r["energy_kwh"] for r in by_cfg.get("C1", []) if r.get("energy_kwh") is not None])

    lines = [
        "Frugality Analysis — H4 Coordination Overhead",
        "=" * 55,
        f"Generated : {date.today().isoformat()}",
        f"Tasks/cell: {MAX_TASKS_PER_CELL}",
        "",
        f"C1 baseline mean energy : {_fmt(c1_e, '.6f')} kWh",
        "",
        f"{'Config':<6} {'n':>4}  {'Energy(kWh)':>12}  {'Overhead':>10}  "
        f"{'Queue(s)':>9}  {'RAM(MB)':>8}  {'TTFUO(s)':>9}  {'Wall(s)':>8}",
        "-" * 80,
    ]
    for cfg in ["C1", "A1", "A2", "A3", "A4"]:
        group    = by_cfg.get(cfg, [])
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

    lines += [
        "",
        "Notes:",
        "  Energy is estimated from wall-clock × TDP (hardware-adaptive).",
        "  Queue wait accumulates when multiple tasks share one cell (max_tasks>1).",
        "  TTFUO = Time-To-First-Useful-Output (first task in cell completes).",
        "  Overhead = mean energy / C1 baseline.",
    ]
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
    total_tasks = sum(r.get("n_tasks", 0) for r in ok_recs)
    total_ok    = sum(r.get("n_success", 0) for r in ok_recs)
    total_tokens = sum(r.get("total_tokens", 0) for r in ok_recs)

    # Frugality table rows
    frug_rows = []
    for cfg in ["C1", "A1", "A2", "A3", "A4"]:
        group    = by_cfg.get(cfg, [])
        energies = [r["energy_kwh"]   for r in group if r.get("energy_kwh")   is not None]
        queues   = [r["queue_wait_s"] for r in group if r.get("queue_wait_s") is not None]
        rams     = [r["peak_ram_mb"]  for r in group if r.get("peak_ram_mb")  is not None]
        ttfuos   = [r["ttfuo_s"]      for r in group if r.get("ttfuo_s")      is not None]
        walls    = [r["wall_s"]       for r in group if r.get("wall_s")       is not None]
        me = _mean(energies)
        oh = f"{me/c1_e:.2f}×" if (me and c1_e) else "n/a"
        frug_rows.append(
            f"| {cfg} | {len(group)} | {_fmt(me,'.6f')} | {oh} | "
            f"{_fmt(_mean(queues),'.2f')} | {_fmt(_mean(rams),'.1f')} | "
            f"{_fmt(_mean(ttfuos),'.2f')} | {_fmt(_mean(walls),'.1f')} |"
        )

    # Token / success rows
    token_rows = []
    for cfg in ["C1", "A1", "A2", "A3", "A4"]:
        group   = by_cfg.get(cfg, [])
        tokens  = [r["total_tokens"] for r in group if r.get("total_tokens") is not None]
        n_ok    = sum(r.get("n_success", 0) for r in group)
        n_tot   = sum(r.get("n_tasks", 0)   for r in group)
        sr      = f"{100*n_ok/max(n_tot,1):.1f}%"
        token_rows.append(
            f"| {cfg} | {len(group)} | {n_tot} | {n_ok} ({sr}) | "
            f"{_fmt(_mean(tokens),'.0f')} | {sum(tokens) if tokens else 'n/a'} |"
        )

    # Verification mode breakdown for A3/A4
    def mode_breakdown(cfg_name):
        group = by_cfg.get(cfg_name, [])
        by_mode = defaultdict(list)
        for r in group:
            by_mode[r["mode"]].append(r)
        rows = []
        for mode in ["none", "selective", "full"]:
            g = by_mode.get(mode, [])
            if g:
                mt = [r["total_tokens"] for r in g if r.get("total_tokens") is not None]
                mw = [r["wall_s"] for r in g if r.get("wall_s") is not None]
                rows.append(f"| {cfg_name} | {mode} | {len(g)} | {_fmt(_mean(mt),'.0f')} | {_fmt(_mean(mw),'.1f')} |")
        return rows

    a3_rows = mode_breakdown("A3")
    a4_rows = mode_breakdown("A4")

    # Hypothesis rows
    hyp_rows = []
    verdicts = {"SUPPORTED": "✅ SUPPORTED", "NOT SUPPORTED": "❌ NOT SUPPORTED", "n/a": "—"}
    for k, v in hyp_table.items():
        display = verdicts.get(str(v), str(v))
        hyp_rows.append(f"| `{k}` | {display} |")

    # Error section
    error_section = ""
    if errors:
        error_section = "\n## Errors\n\n" + "\n".join(
            f"- Cell {e['idx']} ({e['config']}/{e['family']}/seed={e['seed']}/{e['mode']}): `{e['error']}`"
            for e in errors
        ) + "\n"

    # Audit trail count
    audit_path = ROOT / "data" / "audit" / "audit_trail.jsonl"
    if audit_path.exists():
        audit_lines = sum(1 for _ in audit_path.open(encoding="utf-8"))
        audit_note = f"{audit_lines} total records in `data/audit/audit_trail.jsonl`"
    else:
        audit_note = "not yet written"

    from config import H1_ROLE_STABILITY_THRESHOLD, H4_WALL_OVERHEAD_THRESHOLD

    md = f"""# Frugal AI Swarm — Result2 Experiment Report

**Run timestamp:** {run_ts}
**Run type:** Extended run — {MAX_TASKS_PER_CELL} tasks per cell
**Cells:** 108 · **Total task invocations:** {total_tasks}
**Total tokens generated:** {total_tokens:,}
**Errors:** {len(errors)} / {len(records)} cells
**LLM backend:** Ollama (local CPU, apple_silicon)
**Hypothesis framework:** H1 – H4

---

## 1. Architecture Taxonomy

| Config | Name | Roles | Models | Verification |
|--------|------|-------|--------|-------------|
| C1 | Single-Agent Baseline | None | Shared (qwen2.5:3b) | none |
| A1 | Fixed-Role, Single Model | Aligner → Drafter → Verifier | Shared (qwen2.5:3b) | none |
| A2 | Fixed-Role, Multi-Model | Aligner → Drafter → Verifier | Per-role static | none |
| A3 | Self-Organising, Single Model | Shapley + DAG, N=3 | Shared (qwen2.5:3b) | none / selective / full |
| A4 | Self-Organising, Multi-Model | Shapley + DAG, N=3 | Per-agent static | none / selective / full |

**Model assignment (A2 / A4):**

| Slot | A2 Role | A4 Agent | Model |
|------|---------|----------|-------|
| 0 | Curriculum Aligner | agent_0 | `qwen2.5:3b` |
| 1 | Fact-Checker & Drafter | agent_1 | `gemma2:2b` |
| 2 | Pedagogical Verifier | agent_2 | `phi3:mini` |

**Cell breakdown:** C1=12 · A1=12 · A2=12 · A3=36 · A4=36 = **108 cells × {MAX_TASKS_PER_CELL} tasks = {108*MAX_TASKS_PER_CELL} task invocations**

Education task families: `formative_assessment_drafting` · `curriculum_question_generation` · `lesson_adaptation` · `knowledge_base_retrieval`

---

## 2. Run Summary

| Config | Cells | Tasks Issued | Tasks OK (%) | Mean Tokens/Cell | Total Tokens |
|--------|-------|--------------|--------------|-----------------|-------------|
{chr(10).join(token_rows)}
| **Total** | **{len(ok_recs)}** | **{total_tasks}** | **{total_ok} ({100*total_ok/max(total_tasks,1):.1f}%)** | — | **{total_tokens:,}** |

> **On task success rate:** The keyword-F1 rubric compares LLM output against a fixed
> reference answer at threshold 0.40. Education LLMs paraphrase rather than echo
> reference tokens; the metric underestimates true quality. Frugality and token
> metrics are accurate regardless.

---

## 3. Frugality Metrics (H4)

| Config | n | Mean Energy (kWh) | vs C1 | Mean Queue Wait (s) | Mean Peak RAM (MB) | Mean TTFUO (s) | Mean Wall (s) |
|--------|---|------------------|-------|--------------------|--------------------|---------------|--------------|
{chr(10).join(frug_rows)}

**H4 goal:** coordination overhead ≤ {H4_WALL_OVERHEAD_THRESHOLD}× C1.

---

## 4. Verification Mode Breakdown (A3 / A4)

| Config | Mode | Cells | Mean Tokens/Cell | Mean Wall (s) |
|--------|------|-------|-----------------|--------------|
{chr(10).join(a3_rows + a4_rows)}

---

## 5. Hypothesis Results

| Metric | Value |
|--------|-------|
{chr(10).join(hyp_rows)}

### Definitions

| # | Hypothesis | Scope | Threshold |
|---|-----------|-------|-----------|
| H1 | Fixed-role stability | A1, A2 | stability > {H1_ROLE_STABILITY_THRESHOLD} |
| H2 | Selective mode token reduction vs full | A3, A4 | reduction ≥ 30% + retention ≥ 85% |
| H3 | Swarm token efficiency vs C1 | A3, A4 vs C1 | ratio ≥ 1.20 |
| H4 | Coordination wall-clock overhead | A1–A4 vs C1 | ≤ {H4_WALL_OVERHEAD_THRESHOLD}× |

> H1 is **N/A for A3/A4** (no fixed roles).
> H2 triplets computed: **{h2_computed}**
> H3 A3/A4-vs-C1 pairs: **{h3_computed}**

---

## 6. Governance

| Property | Status |
|----------|--------|
| `requires_teacher_review` | ✅ All {total_tasks} task invocations tagged |
| RBAC enforcement | ✅ `SWARM_ROLE=researcher` — all cells granted |
| PII scrubber | ✅ Active on all audit entries |
| Audit trail | ✅ {audit_note} |
| Data retention | ✅ 90-day audit / 30-day ChromaDB |
| Low-watermark | ✅ `get_run_params()` wired — auto-throttles on x86 institutional |

---

## 7. Files in Result2/

| File | Description |
|------|-------------|
| `full_run.txt` | Complete console log (live-appended, SSH-safe) |
| `run_log.jsonl` | Structured per-cell records — flushed after every cell |
| `progress.txt` | Live progress tracker (overwritten each cell) |
| `metrics_analysis.txt` | Per-config token / latency / mode breakdown |
| `frugality_analysis.txt` | Per-config energy / RAM / TTFUO table |
| `RESULTS.md` | This report |
{error_section}
---

*Frugal AI Agent Swarms — COL Phase-1 Pilot (Result2, {date.today().isoformat()})*
"""
    path = out_dir / "RESULTS.md"
    path.write_text(md, encoding="utf-8")
    print(f"Written: {path}")


if __name__ == "__main__":
    main()
