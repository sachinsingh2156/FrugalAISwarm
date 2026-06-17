"""
Experiment runner — A1-A4 taxonomy (education-focused).

Executes all evaluation cells (C1, A1, A2, A3, A4), handling graph dispatch,
token-budget measurement, and metric collection.

Each cell:
  1. RBAC check + hardware snapshot.
  2. Load education-family tasks.
  3. Build and invoke the LangGraph swarm graph.
  4. Collect frugality metrics (H4).
  5. Log to MLflow and ChromaDB.

Dispatch table:
  C1  — build_c3_graph(N=1)         single-agent baseline
  A1  — build_c2_edu_graph()         fixed-role, single model
  A2  — build_a2_graph()             fixed-role, multi-model
  A3  — build_c3_graph(N=3)         self-organising, single model
  A4  — build_a4_graph()             self-organising, multi-model

Hypothesis anchoring:
  H1 (role stability): A1/A2 only — fixed-role configs trivially achieve
      stability=1.0; A3/A4 have no fixed roles so H1 is noted as N/A.
  H2 (verification overhead): A3/A4 across verification modes.
  H3 (token efficiency): A3/A4-none vs C1.
  H4 (coordination overhead): all of A1-A4 vs C1.
"""
from __future__ import annotations

import json
import uuid

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import VerificationMode
from frugal_swarm.coordination.state import Task, make_dashboard_state
from frugal_swarm.corpus.loader import load_family, EDUCATION_FAMILIES
from frugal_swarm.experiments.configurations import (
    ExperimentConfig, enumerate_cells,
)
from frugal_swarm.experiments.metrics import (
    RunMetrics, compute_run_metrics, compute_h2, compute_h3, hypothesis_table,
)
from frugal_swarm.graph.swarm_graph import (
    build_c3_graph,
    build_c2_edu_graph,
    build_a2_graph,
    build_a4_graph,
)
from frugal_swarm.coordination.role_monitor import RoleMonitor
from frugal_swarm.memory.chroma_store import ChromaStore
from frugal_swarm.memory.mlflow_tracker import MLflowTracker
from frugal_swarm.model.ollama_client import OllamaClient
from frugal_swarm.metrics.frugality import FrugalityCollector, log_frugality_to_mlflow
from frugal_swarm.metrics.hardware import get_hardware_snapshot, get_run_params
from frugal_swarm.governance.rbac import get_current_role, is_allowed
from frugal_swarm.governance.audit_trail import log_action

console = Console()

# Self-organising config names (H2/H3/H4 apply; H1 is N/A).
_SELF_ORG_CONFIGS = {"A3", "A4"}
# Fixed-role config names (H1 applies trivially; H2 is N/A since they run one mode).
_FIXED_ROLE_CONFIGS = {"A1", "A2"}


def _agent_ids(n: int) -> list[str]:
    return [f"agent_{i}" for i in range(n)]


def _invoke_graph_for_task(
    compiled_graph,
    task: Task,
    agent_ids: list[str],
    run_id: str,
    config_name: str,
    verification_mode: str,
) -> dict:
    """Run one task through a compiled LangGraph and return the final state."""
    thread_id = f"{run_id}_{task.task_id}"
    initial_state = make_dashboard_state(
        task=task,
        agent_ids=agent_ids,
        run_id=thread_id,
        verification_mode=verification_mode,
        config_name=config_name,
    )
    lg_config = {"configurable": {"thread_id": thread_id}}
    return compiled_graph.invoke(initial_state, config=lg_config)


def _complete_task(task: Task, final_state: dict) -> Task:
    task.status = "done"
    task.result = final_state.get("final_response", "")
    task.tokens_used = final_state.get("total_tokens", 0)
    return task


def _run_graph_batch(
    compiled_graph,
    tasks: list[Task],
    agent_ids: list[str],
    run_id: str,
    config_name: str,
    verification_mode: str,
    frugality: FrugalityCollector | None = None,
    output_jsonl: str | None = None,
    seed: int | None = None,
) -> list[Task]:
    completed: list[Task] = []
    for task in tasks:
        t_task_start = __import__("time").perf_counter()
        final_state = _invoke_graph_for_task(
            compiled_graph,
            task,
            agent_ids,
            run_id,
            config_name,
            verification_mode,
        )
        done = _complete_task(task, final_state)
        completed.append(done)
        log_action(
            session_id=run_id,
            action_type="task_completed",
            actor="system",
            task_id=task.task_id,
            payload={
                "family": task.family,
                "tokens_used": done.tokens_used,
                "status": done.status,
                "verification_mode": verification_mode,
            },
            config_name=config_name,
        )
        if output_jsonl is not None:
            try:
                with open(output_jsonl, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps({
                        "run_id": run_id,
                        "config": config_name,
                        "family": task.family,
                        "seed": seed,
                        "mode": verification_mode,
                        "task_num": len(completed),
                        "task_id": task.task_id,
                        "prompt": task.prompt,
                        "response": done.result or "",
                        "tokens": done.tokens_used,
                        "status": done.status,
                    }) + "\n")
            except Exception:
                pass
        if frugality is not None:
            frugality.record_first_output()
            if len(completed) > 1:
                frugality.add_queue_wait(__import__("time").perf_counter() - t_task_start)
    return completed


def run_cell(
    config: ExperimentConfig,
    family: str,
    seed: int,
    mode: VerificationMode,
    client: OllamaClient,
    chroma: ChromaStore,
    mlflow_tracker: MLflowTracker,
    max_tasks: int | None = None,
    role_mon: RoleMonitor | None = None,
    output_jsonl: str | None = None,
) -> RunMetrics:
    """Run one experimental cell and return its RunMetrics."""
    run_id = f"{config.name}_{family}_{seed}_{mode.value}_{uuid.uuid4().hex[:6]}"

    hw = get_hardware_snapshot()
    hw_params = get_run_params()
    role = get_current_role()
    granted = is_allowed(role, "submit_task")
    role_str = role.value if hasattr(role, "value") else str(role)

    console.print(
        f"  [cyan]RUN[/cyan] {run_id} | model={client.model} | "
        f"hw={hw.platform_tag}{' (low-watermark)' if hw.low_watermark_mode else ''} | "
        f"role={role_str} ({'granted' if granted else 'DENIED'})"
    )
    if not granted:
        raise PermissionError(
            f"RBAC: role '{role_str}' is not permitted to submit_task. "
            f"Set SWARM_ROLE=researcher (or admin) to run experiments."
        )

    log_action(
        session_id=run_id,
        action_type="run_started",
        actor=role_str,
        task_id="N/A",
        payload={
            "config": config.name,
            "family": family,
            "seed": seed,
            "mode": mode.value,
            "low_watermark": hw.low_watermark_mode,
            "hw_params": hw_params,
        },
        config_name=config.name,
    )

    tasks = load_family(family, seed=seed)
    if max_tasks:
        tasks = tasks[:max_tasks]

    chroma.store_tasks(tasks)

    agent_ids = _agent_ids(config.swarm_size)

    run_params = {
        "config": config.name,
        "family": family,
        "seed": seed,
        "mode": mode.value,
        "swarm_size": config.swarm_size,
        "model": client.model,
        "platform_tag": hw.platform_tag,
        "low_watermark_mode": hw.low_watermark_mode,
        "rbac_role": role_str,
        "hw_max_tokens": hw_params["max_tokens"],
        "hw_max_rounds": hw_params["max_rounds"],
    }
    with mlflow_tracker.run(run_name=run_id, params=run_params) as tracker:
        fc = FrugalityCollector(tdp_watts=300.0 if hw.platform_tag == "x86_institutional" else None)
        fc.start()

        if config.use_roles:
            metrics = _run_fixed_role(
                config, tasks, agent_ids, client,
                run_id, family, seed, mode,
                frugality=fc,
                max_tokens=hw_params["max_tokens"],
                output_jsonl=output_jsonl,
            )
        else:
            metrics = _run_swarm(
                config, tasks, agent_ids, client,
                run_id, family, seed, mode,
                role_mon=role_mon,
                frugality=fc,
                max_tokens=hw_params["max_tokens"],
                max_rounds=hw_params["max_rounds"],
                output_jsonl=output_jsonl,
            )

        snap = fc.finish()

        metrics.energy_kwh_est = snap.energy_kwh_est
        metrics.queue_wait_s = snap.queue_wait_s
        metrics.ttfuo_s = snap.ttfuo_s
        metrics.peak_ram_mb = snap.peak_ram_mb
        metrics.platform_tag = hw.platform_tag
        metrics.low_watermark_mode = hw.low_watermark_mode
        metrics.rbac_role = role_str
        metrics.rbac_granted = granted
        metrics.requires_teacher_review = family in EDUCATION_FAMILIES

        tracker.log_metrics({
            "task_success_rate":    metrics.task_success_rate,
            "n_tasks":              float(metrics.n_tasks),
            "n_success":            float(metrics.n_success),
            "mean_tokens_per_task": metrics.mean_tokens_per_task,
            "mean_latency_per_task": metrics.mean_latency_per_task,
            "total_tokens":         float(metrics.total_tokens),
        })
        tracker.log_hypothesis_results(
            h1_stability=metrics.mean_role_stability,
            h1_passed=metrics.h1_passed,
            h2_token_reduction=metrics.verification_token_overhead,
            h2_reliability_retention=metrics.reliability_gain_retention,
            h2_passed=metrics.h2_passed,
            h3_efficiency_ratio=metrics.token_efficiency_ratio,
            h3_passed=metrics.h3_passed,
        )
        log_frugality_to_mlflow(snap)

        console.print(
            f"    [dim]frugality: energy={snap.energy_kwh_est:.6f} kWh | "
            f"queue_wait={snap.queue_wait_s:.2f}s | "
            f"ttfuo={snap.ttfuo_s if snap.ttfuo_s is not None else 'n/a'} | "
            f"peak_ram={snap.peak_ram_mb if snap.peak_ram_mb is not None else 'n/a'} MB[/dim]"
        )

    log_action(
        session_id=run_id,
        action_type="run_complete",
        actor="system",
        task_id="N/A",
        payload={
            "n_tasks": metrics.n_tasks,
            "n_success": metrics.n_success,
            "total_tokens": metrics.total_tokens,
            "energy_kwh_est": snap.energy_kwh_est,
            "peak_ram_mb": snap.peak_ram_mb,
        },
        config_name=config.name,
    )

    return metrics


def _run_fixed_role(
    config: ExperimentConfig,
    tasks: list[Task],
    agent_ids: list[str],
    client: OllamaClient,
    run_id: str,
    family: str,
    seed: int,
    mode: VerificationMode,
    frugality: FrugalityCollector | None = None,
    max_tokens: int = 500,
    output_jsonl: str | None = None,
) -> RunMetrics:
    """Run A1 (single-model) or A2 (multi-model) fixed-role pipeline."""
    role_agent_ids = agent_ids[:3]
    if config.model_assignment:
        compiled = build_a2_graph(
            agent_ids=role_agent_ids,
            client=client,
            model_assignment=config.model_assignment,
            max_tokens=max_tokens,
        )
    else:
        compiled = build_c2_edu_graph(
            agent_ids=role_agent_ids,
            client=client,
            max_tokens=max_tokens,
        )

    completed = _run_graph_batch(
        compiled,
        tasks,
        role_agent_ids,
        run_id,
        config.name,
        mode.value,
        frugality=frugality,
        output_jsonl=output_jsonl,
        seed=seed,
    )

    # Fixed-role configs trivially achieve role stability = 1.0 (roles never change).
    return compute_run_metrics(
        run_id=run_id,
        config_name=config.name,
        family=family,
        seed=seed,
        verification_mode=mode.value,
        swarm_size=config.swarm_size,
        completed_tasks=completed,
        role_stability=1.0,
        specialisation_entropy=0.0,
    )


def _run_swarm(
    config: ExperimentConfig,
    tasks: list[Task],
    agent_ids: list[str],
    client: OllamaClient,
    run_id: str,
    family: str,
    seed: int,
    mode: VerificationMode,
    role_mon: RoleMonitor | None = None,
    frugality: FrugalityCollector | None = None,
    max_tokens: int = 400,
    max_rounds: int | None = None,
    output_jsonl: str | None = None,
) -> RunMetrics:
    """Run C1, A3 (single-model), or A4 (multi-model) self-organising swarm."""
    if config.model_assignment:
        compiled, _shapley, role_mon = build_a4_graph(
            agent_ids=agent_ids,
            client=client,
            model_assignment=config.model_assignment,
            mode=mode,
            role_mon=role_mon,
            max_tokens=max_tokens,
            max_rounds=max_rounds,
        )
    else:
        compiled, _shapley, role_mon = build_c3_graph(
            agent_ids=agent_ids,
            client=client,
            mode=mode,
            role_mon=role_mon,
            max_tokens=max_tokens,
            max_rounds=max_rounds,
        )

    completed = _run_graph_batch(
        compiled,
        tasks,
        agent_ids,
        run_id,
        config.name,
        mode.value,
        frugality=frugality,
        output_jsonl=output_jsonl,
        seed=seed,
    )
    role_summary = role_mon.summary()

    return compute_run_metrics(
        run_id=run_id,
        config_name=config.name,
        family=family,
        seed=seed,
        verification_mode=mode.value,
        swarm_size=config.swarm_size,
        completed_tasks=completed,
        role_stability=role_summary.get("mean_role_stability", 0.0),
        specialisation_entropy=role_summary.get("specialisation_entropy", 0.0),
    )


def run_all(
    configs: list[ExperimentConfig] | None = None,
    client: OllamaClient | None = None,
    max_tasks_per_cell: int | None = None,
) -> list[RunMetrics]:
    """
    Run all experimental cells and return the full list of RunMetrics.
    Pass max_tasks_per_cell=1 for a quick smoke test.
    """
    from collections import defaultdict
    from config import ROLE_STABILITY_WINDOW, H1_ROLE_STABILITY_THRESHOLD

    if client is None:
        client = OllamaClient()
    chroma = ChromaStore()
    mlflow_tracker = MLflowTracker()

    active_configs = configs or []
    cells = enumerate_cells(active_configs)
    all_metrics: list[RunMetrics] = []

    # Shared role monitors for self-organising configs (A3/A4) so Shapley data
    # accumulates across all family/seed/mode cells.
    shared_role_monitors: dict[str, RoleMonitor] = {}
    for cfg in active_configs:
        if cfg.name in _SELF_ORG_CONFIGS and cfg.name not in shared_role_monitors:
            shared_role_monitors[cfg.name] = RoleMonitor(
                _agent_ids(cfg.swarm_size), window_size=ROLE_STABILITY_WINDOW
            )

    console.rule("[bold blue]Frugal AI Swarm — Education Experiment Run[/bold blue]")
    console.print(f"Total cells to run: {len(cells)}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_bar = progress.add_task("Running cells...", total=len(cells))

        for cell in cells:
            cfg = cell["config"]
            family = cell["family"]
            seed = cell["seed"]
            mode = cell["verification_mode"]
            progress.update(
                task_bar,
                description=f"{cfg.name} | {family} | seed={seed} | {mode.value}",
            )
            try:
                m = run_cell(
                    config=cfg,
                    family=family,
                    seed=seed,
                    mode=mode,
                    client=client,
                    chroma=chroma,
                    mlflow_tracker=mlflow_tracker,
                    max_tasks=max_tasks_per_cell,
                    role_mon=shared_role_monitors.get(cfg.name),
                )
                all_metrics.append(m)
            except Exception as exc:
                console.print(f"[red]ERROR in cell {cfg.name}/{family}/{seed}/{mode.value}: {exc}[/red]")
            progress.advance(task_bar)

    # ── H1: A1/A2 role stability is trivially 1.0 (already set in _run_fixed_role).
    # A3/A4 share accumulated role monitor data across cells.
    for cfg_name, role_mon in shared_role_monitors.items():
        summary = role_mon.summary()
        final_stability = summary.get("mean_role_stability", 0.0)
        final_entropy = summary.get("specialisation_entropy", 0.0)
        for m in all_metrics:
            if m.config_name == cfg_name:
                m.mean_role_stability = final_stability
                m.specialisation_entropy = final_entropy
                # H1 is N/A for self-organising configs; mark as False/N/A.
                m.h1_passed = None  # type: ignore[assignment]

    # ── H2: verification mode comparison for A3/A4 ───────────────────────────
    swarm_by_cell: dict = defaultdict(dict)
    for m in all_metrics:
        if m.config_name in _SELF_ORG_CONFIGS:
            key = (m.config_name, m.family, m.seed)
            swarm_by_cell[key][m.verification_mode] = m

    h2_computed = 0
    for key, mode_runs in swarm_by_cell.items():
        none_run = mode_runs.get("none")
        full_run = mode_runs.get("full")
        selective_run = mode_runs.get("selective")
        if none_run and full_run and selective_run:
            h2_result = compute_h2(none_run, full_run, selective_run)
            for k, v in h2_result.items():
                setattr(selective_run, k, v)
            h2_computed += 1

    console.print(f"  H2 computed for {h2_computed} (config, family, seed) triplets")

    # ── H3: pair each A3/A4-none run with matching C1 run ────────────────────
    c1_by_cell: dict = {}
    for m in all_metrics:
        if m.config_name == "C1" and m.verification_mode == "none":
            c1_by_cell[(m.family, m.seed)] = m

    h3_computed = 0
    for m in all_metrics:
        if m.config_name in _SELF_ORG_CONFIGS and m.verification_mode == "none":
            c1_run = c1_by_cell.get((m.family, m.seed))
            if c1_run:
                h3_result = compute_h3(m, c1_run)
                for k, v in h3_result.items():
                    setattr(m, k, v)
                h3_computed += 1

    console.print(f"  H3 computed for {h3_computed} A3/A4-vs-C1 pairs")

    table = hypothesis_table(all_metrics)
    console.rule("[bold green]Hypothesis Verification Table[/bold green]")
    for k, v in table.items():
        console.print(f"  {k}: {v}")

    return all_metrics
