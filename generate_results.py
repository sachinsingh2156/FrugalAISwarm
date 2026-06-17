"""
Post-processing script: parses the experiment run log and writes
  results/metrics_analysis.txt
  results/frugality_analysis.txt
  results/RESULTS.md

Run after the experiment completes:
    python generate_results.py [--input results/full_run.txt] [--out-dir results/]

Config taxonomy: C1, A1, A2, A3, A4
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent

# ── Regex patterns ────────────────────────────────────────────────────────────

RUN_PAT = re.compile(
    r"RUN (?P<run_id>\S+) \| model=(?P<model>\S+)"
    r" \| hw=(?P<hw>\S+)"
    r" \| role=(?P<role>\S+) \((?P<rbac>granted|DENIED)\)"
)
FRUG_PAT = re.compile(
    r"frugality: energy=(?P<energy>[\d.]+) kWh"
    r" \| queue_wait=(?P<queue>[\d.]+)s"
    r" \| ttfuo=(?P<ttfuo>[\d.na/]+)"
    r" \| peak_ram=(?P<ram>[\d.na/]+) MB"
)
H2_PAT = re.compile(r"H2 computed for (\d+)")
H3_PAT = re.compile(r"H3 computed for (\d+)")
HYP_PAT = re.compile(r"^\s+(H[1234]_\w+): (.+)$")


def _join_wrapped_lines(raw: str) -> str:
    """
    Rich wraps long console lines at ~80 chars.  Re-join continuation lines so
    regexes can match the original logical line.
    """
    lines = raw.splitlines()
    joined: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if joined and (
            stripped.startswith("role=")
            or stripped.startswith("| role=")
            or stripped.startswith("hw=")
            or stripped.startswith("| peak_ram=")
        ):
            joined[-1] = joined[-1].rstrip() + " " + stripped
        else:
            joined.append(line)
    return "\n".join(joined)


def parse_run_id(run_id: str):
    """
    Parse config / family / seed / mode from run IDs like:
      A3_formative_assessment_drafting_42_full_abc123
      C1_knowledge_base_retrieval_7_none_def456
      A2_curriculum_question_generation_137_none_789abc
    """
    parts = run_id.split("_")
    parts = parts[:-1]       # drop hex suffix

    # Config is always the first token (C1, A1, A2, A3, A4)
    config = parts[0]
    rest = parts[1:]

    # mode is the last token of rest
    mode = rest[-1] if rest else "none"
    # seed is second-to-last
    seed = rest[-2] if len(rest) >= 2 else "0"
    # family is everything before seed
    family = "_".join(rest[:-2]) if len(rest) > 2 else "_".join(rest[:-1])
    return config, family, seed, mode


def parse_log(raw: str) -> dict:
    text = _join_wrapped_lines(raw)

    runs: list[dict] = []
    current: dict | None = None

    h2_total = 0
    h3_total = 0
    hypothesis_lines: list[str] = []

    for line in text.splitlines():
        m = RUN_PAT.search(line)
        if m:
            if current:
                runs.append(current)
            config, family, seed, mode = parse_run_id(m.group("run_id"))
            current = {
                "run_id":   m.group("run_id"),
                "config":   config,
                "family":   family,
                "seed":     seed,
                "mode":     mode,
                "model":    m.group("model"),
                "hw":       m.group("hw"),
                "rbac":     m.group("rbac"),
                "energy":   None,
                "queue":    None,
                "ttfuo":    None,
                "ram":      None,
            }
            continue

        f = FRUG_PAT.search(line)
        if f and current:
            def _f(v: str) -> float | None:
                try:
                    return float(v)
                except ValueError:
                    return None

            current["energy"] = _f(f.group("energy"))
            current["queue"]  = _f(f.group("queue"))
            current["ttfuo"]  = _f(f.group("ttfuo"))
            current["ram"]    = _f(f.group("ram"))
            continue

        h2m = H2_PAT.search(line)
        if h2m:
            h2_total = int(h2m.group(1))
            continue

        h3m = H3_PAT.search(line)
        if h3m:
            h3_total = int(h3m.group(1))
            continue

        hm = HYP_PAT.match(line)
        if hm:
            hypothesis_lines.append(f"{hm.group(1)}: {hm.group(2)}")

    if current:
        runs.append(current)

    return {
        "runs": runs,
        "h2_total": h2_total,
        "h3_total": h3_total,
        "hypothesis_lines": hypothesis_lines,
    }


def metrics_summary(runs: list[dict]) -> str:
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_config[r["config"]].append(r)

    lines = ["Metrics Summary by Configuration", "=" * 40, ""]
    for cfg in ["C1", "A1", "A2", "A3", "A4"]:
        group = by_config.get(cfg, [])
        if not group:
            continue
        energies = [r["energy"] for r in group if r["energy"] is not None]
        queues   = [r["queue"]  for r in group if r["queue"]  is not None]
        rams     = [r["ram"]    for r in group if r["ram"]    is not None]
        ttfuos   = [r["ttfuo"]  for r in group if r["ttfuo"]  is not None]
        granted  = sum(1 for r in group if r["rbac"] == "granted")

        lines.append(f"[{cfg}]  n={len(group)}  rbac_granted={granted}/{len(group)}")
        if energies:
            lines.append(f"  energy_kWh : mean={sum(energies)/len(energies):.6f}  "
                         f"total={sum(energies):.6f}")
        if queues:
            lines.append(f"  queue_wait : mean={sum(queues)/len(queues):.2f}s")
        if rams:
            lines.append(f"  peak_ram   : mean={sum(rams)/len(rams):.1f} MB")
        if ttfuos:
            lines.append(f"  ttfuo      : mean={sum(ttfuos)/len(ttfuos):.2f}s")
        lines.append("")

    return "\n".join(lines)


def frugality_summary(runs: list[dict]) -> str:
    """Cross-config frugality comparison (H4 overhead proxy)."""
    c1 = [r for r in runs if r["config"] == "C1"]
    swarm = [r for r in runs if r["config"] in {"A1", "A2", "A3", "A4"}]

    lines = ["Frugality Analysis (H4 — Coordination Overhead)", "=" * 50, ""]

    def mean_energy(group: list[dict]) -> float | None:
        vals = [r["energy"] for r in group if r["energy"] is not None]
        return sum(vals) / len(vals) if vals else None

    c1_e   = mean_energy(c1)
    swarm_e = mean_energy(swarm)
    if c1_e and swarm_e:
        overhead = swarm_e / c1_e
        lines.append(f"Mean energy  C1={c1_e:.6f} kWh  swarm={swarm_e:.6f} kWh  "
                     f"overhead={overhead:.2f}×")
    else:
        lines.append("Energy data incomplete — overhead not computed.")

    lines.append("")
    lines.append("Per-config mean energy (kWh):")
    by_cfg: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if r["energy"] is not None:
            by_cfg[r["config"]].append(r["energy"])
    for cfg in ["C1", "A1", "A2", "A3", "A4"]:
        vals = by_cfg.get(cfg, [])
        if vals:
            lines.append(f"  {cfg}: {sum(vals)/len(vals):.6f} kWh (n={len(vals)})")

    return "\n".join(lines)


def results_md(parsed: dict) -> str:
    runs = parsed["runs"]
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_config[r["config"]].append(r)

    config_labels = {
        "C1":  "C1 — Single-Agent Baseline",
        "A1":  "A1 — Fixed-Role, Single Model",
        "A2":  "A2 — Fixed-Role, Multi-Model",
        "A3":  "A3 — Self-Organising, Single Model",
        "A4":  "A4 — Self-Organising, Multi-Model",
    }

    lines = [
        f"# Frugal AI Swarm — Experiment Results",
        f"",
        f"Generated: {date.today().isoformat()}",
        f"",
        f"## Configuration Taxonomy",
        f"",
        f"| Config | Architecture | Roles | Models |",
        f"|--------|-------------|-------|--------|",
        f"| C1  | Single-agent baseline | None | Shared |",
        f"| A1  | Fixed-role pipeline   | Aligner → Drafter → Verifier | Shared |",
        f"| A2  | Fixed-role pipeline   | Aligner → Drafter → Verifier | Per-role static |",
        f"| A3  | Self-organising swarm | Shapley + DAG, N=3 | Shared |",
        f"| A4  | Self-organising swarm | Shapley + DAG, N=3 | Per-agent static |",
        f"",
        f"All configurations run on **education task families only**:",
        f"formative_assessment_drafting, curriculum_question_generation,",
        f"lesson_adaptation, knowledge_base_retrieval.",
        f"",
        f"## Run Summary",
        f"",
        f"| Config | Cells | RBAC Granted |",
        f"|--------|-------|-------------|",
    ]

    for cfg in ["C1", "A1", "A2", "A3", "A4"]:
        group = by_config.get(cfg, [])
        granted = sum(1 for r in group if r["rbac"] == "granted")
        lines.append(f"| {cfg} | {len(group)} | {granted}/{len(group)} |")

    lines += [
        f"",
        f"## Frugality (H4)",
        f"",
        f"| Config | Mean Energy (kWh) | Mean Queue Wait (s) | Mean Peak RAM (MB) |",
        f"|--------|------------------|--------------------|--------------------|",
    ]

    for cfg in ["C1", "A1", "A2", "A3", "A4"]:
        group = by_config.get(cfg, [])
        energies = [r["energy"] for r in group if r["energy"] is not None]
        queues   = [r["queue"]  for r in group if r["queue"]  is not None]
        rams     = [r["ram"]    for r in group if r["ram"]    is not None]
        e_str = f"{sum(energies)/len(energies):.6f}" if energies else "n/a"
        q_str = f"{sum(queues)/len(queues):.2f}"     if queues   else "n/a"
        r_str = f"{sum(rams)/len(rams):.1f}"         if rams     else "n/a"
        lines.append(f"| {cfg} | {e_str} | {q_str} | {r_str} |")

    lines += [
        f"",
        f"## Hypothesis Results",
        f"",
        f"| Hypothesis | Scope | Status |",
        f"|-----------|-------|--------|",
        f"| H1: Role stability > 70% | A1, A2 | See MLflow |",
        f"| H2: Verification token reduction | A3, A4 | See MLflow |",
        f"| H3: Token efficiency ≥ 1.2× vs C1 | A3, A4 vs C1 | See MLflow |",
        f"| H4: Wall-clock overhead ≤ 3× vs C1 | A1–A4 vs C1 | See MLflow |",
        f"",
        f"> Note: H1 is **not applicable** to A3/A4 — self-organising agents have",
        f"> no fixed roles, so role stability is measured as an exploratory metric",
        f"> only (not a hypothesis test).",
        f"",
        f"## H2/H3 Cross-Mode Triplets",
        f"",
        f"H2 triplets computed: **{parsed['h2_total']}**",
        f"H3 A3/A4-vs-C1 pairs computed: **{parsed['h3_total']}**",
        f"",
        f"## Governance",
        f"",
        f"- All tasks tagged `requires_teacher_review=True` (education families).",
        f"- RBAC enforced via `SWARM_ROLE` env variable (`researcher` or `admin`).",
        f"- Frugality collector active for all cells (energy / queue_wait / TTFUO / RAM).",
    ]

    if parsed["hypothesis_lines"]:
        lines += [f"", f"## Raw Hypothesis Summary Lines", f""]
        lines += [f"    {l}" for l in parsed["hypothesis_lines"]]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate results from experiment run log.")
    parser.add_argument(
        "--input", default=None,
        help="Path to run log file (default: results/full_run.txt)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: results/)",
    )
    args = parser.parse_args()

    results_dir = Path(args.out_dir) if args.out_dir else ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    input_file = Path(args.input) if args.input else results_dir / "full_run.txt"
    if not input_file.exists():
        print(f"Error: input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    raw = input_file.read_text(encoding="utf-8", errors="replace")
    parsed = parse_log(raw)
    runs = parsed["runs"]
    print(f"Parsed {len(runs)} runs from {input_file}")

    metrics_path = results_dir / "metrics_analysis.txt"
    metrics_path.write_text(metrics_summary(runs), encoding="utf-8")
    print(f"Written: {metrics_path}")

    frugality_path = results_dir / "frugality_analysis.txt"
    frugality_path.write_text(frugality_summary(runs), encoding="utf-8")
    print(f"Written: {frugality_path}")

    md_path = results_dir / "RESULTS.md"
    md_path.write_text(results_md(parsed), encoding="utf-8")
    print(f"Written: {md_path}")


if __name__ == "__main__":
    main()
