"""
Dashboard Runner — A1-A4 taxonomy, all backed by LangGraph StateGraph.

ARCHITECTURE
------------
Each run_*() function is a Python generator (for Flask SSE), delegating ALL
execution to a compiled LangGraph graph via graph.stream().

The bridge:
  - Each LangGraph node returns {"events": [event_dict, ...], ...state_updates}
  - stream(state, config, stream_mode="updates") yields {node_name: delta}
  - This runner reads delta.get("events", []) and yields each event to SSE

ENTRY POINTS
------------
run_a1(task, agent_ids, client)                             — fixed-role, single model
run_a2(task, agent_ids, client, model_assignment)           — fixed-role, multi-model
run_a3(task, agent_ids, client, mode)                       — self-organising, single model
run_a4(task, agent_ids, client, model_assignment, mode)     — self-organising, multi-model
run_c1(task, agent_ids, client)                             — single-agent baseline
"""
from __future__ import annotations

import time
import uuid
from typing import Generator

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import VerificationMode, MODEL_POOL
from frugal_swarm.model.ollama_client import OllamaClient
from frugal_swarm.coordination.state import Task, make_dashboard_state, make_event
from frugal_swarm.graph.swarm_graph import (
    build_c1_graph,
    build_c3_graph,
    build_c2_edu_graph,
    build_a2_graph,
    build_a4_graph,
)


def _ev(event_type: str, **data) -> dict:
    return {"type": event_type, "ts": time.time(), **data}


def _stream_graph(compiled_graph, initial_state: dict, thread_id: str):
    """
    Yield every event emitted by the LangGraph graph.

    stream_mode="updates" delivers {node_name: delta} after each node.
    We pull delta["events"] and yield each event in order.
    """
    lg_config = {"configurable": {"thread_id": thread_id}}
    for chunk in compiled_graph.stream(initial_state, config=lg_config,
                                        stream_mode="updates"):
        for _node_name, delta in chunk.items():
            for event in delta.get("events", []):
                if "ts" not in event:
                    event["ts"] = time.time()
                yield event


# ═══════════════════════════════════════════════════════════════════════════════
# C1 — Single-Agent Baseline
# ═══════════════════════════════════════════════════════════════════════════════

def run_c1(
    task: Task,
    agent_ids: list[str],
    client: OllamaClient,
) -> Generator[dict, None, None]:
    """C1 single-agent baseline — driven by the C3 graph with N=1."""
    thread_id = str(uuid.uuid4())
    single_agent = agent_ids[:1]

    yield _ev("task_started",
               task_id=task.task_id,
               prompt=task.prompt,
               family=task.family,
               config="C1",
               mode="none",
               swarm_size=1,
               agent_ids=single_agent)

    yield _ev("board_update", open=1, claimed=0, done=0,
               detail="Single-agent baseline — building LangGraph C1")

    try:
        compiled, _, _ = build_c1_graph(agent_ids=single_agent, client=client)

        initial_state = make_dashboard_state(
            task=task,
            agent_ids=single_agent,
            run_id=thread_id,
            verification_mode="none",
            config_name="C1",
        )

        yield from _stream_graph(compiled, initial_state, thread_id)

    except Exception as exc:
        yield _ev("error", message=str(exc), config="C1",
                   detail="LangGraph C1 execution failed")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# A1 — Fixed-Role, Single Model
# ═══════════════════════════════════════════════════════════════════════════════

def run_a1(
    task: Task,
    agent_ids: list[str],
    client: OllamaClient,
) -> Generator[dict, None, None]:
    """
    A1 education fixed-role pipeline, single shared model.
    Graph: Aligner → Fact-Checker/Drafter → Pedagogical Verifier → END
    """
    thread_id = str(uuid.uuid4())

    yield _ev("task_started",
               task_id=task.task_id,
               prompt=task.prompt,
               family=task.family,
               config="A1",
               mode="none",
               swarm_size=min(len(agent_ids), 3),
               agent_ids=agent_ids[:3])

    yield _ev("board_update", open=1, claimed=0, done=0,
               detail="Education pipeline (single model) — building LangGraph A1")

    try:
        compiled = build_c2_edu_graph(agent_ids=agent_ids, client=client)

        initial_state = make_dashboard_state(
            task=task,
            agent_ids=agent_ids[:3],
            run_id=thread_id,
            verification_mode="none",
            config_name="A1",
        )

        yield from _stream_graph(compiled, initial_state, thread_id)

    except Exception as exc:
        yield _ev("error", message=str(exc), config="A1",
                   detail="LangGraph A1 execution failed")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# A2 — Fixed-Role, Multi-Model
# ═══════════════════════════════════════════════════════════════════════════════

def run_a2(
    task: Task,
    agent_ids: list[str],
    client: OllamaClient,
    model_assignment: list[str] | None = None,
) -> Generator[dict, None, None]:
    """
    A2 education fixed-role pipeline, per-role model assignment.
    Aligner(qwen2.5:3b) → Drafter(gemma2:2b) → Verifier(phi3:mini) → END
    """
    thread_id = str(uuid.uuid4())
    models = model_assignment or list(MODEL_POOL)

    yield _ev("task_started",
               task_id=task.task_id,
               prompt=task.prompt,
               family=task.family,
               config="A2",
               mode="none",
               swarm_size=min(len(agent_ids), 3),
               agent_ids=agent_ids[:3],
               model_assignment=models)

    yield _ev("board_update", open=1, claimed=0, done=0,
               detail="Education pipeline (multi-model) — building LangGraph A2")

    try:
        compiled = build_a2_graph(
            agent_ids=agent_ids,
            client=client,
            model_assignment=models,
        )

        initial_state = make_dashboard_state(
            task=task,
            agent_ids=agent_ids[:3],
            run_id=thread_id,
            verification_mode="none",
            config_name="A2",
        )

        yield from _stream_graph(compiled, initial_state, thread_id)

    except Exception as exc:
        yield _ev("error", message=str(exc), config="A2",
                   detail="LangGraph A2 execution failed")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# A3 — Self-Organising, Single Model
# ═══════════════════════════════════════════════════════════════════════════════

def run_a3(
    task: Task,
    agent_ids: list[str],
    client: OllamaClient,
    mode: VerificationMode = VerificationMode.NONE,
) -> Generator[dict, None, None]:
    """
    A3 self-organising swarm, single shared model.
    Graph: agent_0 → agent_1 → agent_2 → coordinator → [loop | finaliser] → END
    Shapley + DAG negotiation over 2 rounds.
    """
    thread_id = str(uuid.uuid4())

    yield _ev("task_started",
               task_id=task.task_id,
               prompt=task.prompt,
               family=task.family,
               config="A3",
               mode=mode.value,
               swarm_size=len(agent_ids),
               agent_ids=agent_ids)

    yield _ev("board_update", open=1, claimed=0, done=0,
               detail="Self-organising swarm (single model) — building LangGraph A3")

    try:
        compiled, _, _ = build_c3_graph(
            agent_ids=agent_ids,
            client=client,
            mode=mode,
        )

        initial_state = make_dashboard_state(
            task=task,
            agent_ids=agent_ids,
            run_id=thread_id,
            verification_mode=mode.value,
            config_name="A3",
        )

        yield from _stream_graph(compiled, initial_state, thread_id)

    except Exception as exc:
        yield _ev("error", message=str(exc), config="A3",
                   detail="LangGraph A3 execution failed")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# A4 — Self-Organising, Multi-Model
# ═══════════════════════════════════════════════════════════════════════════════

def run_a4(
    task: Task,
    agent_ids: list[str],
    client: OllamaClient,
    model_assignment: list[str] | None = None,
    mode: VerificationMode = VerificationMode.NONE,
) -> Generator[dict, None, None]:
    """
    A4 self-organising swarm, per-agent model assignment.
    agent_0(qwen) → agent_1(gemma) → agent_2(phi3) → coordinator → … → END
    Same Shapley + DAG negotiation as A3; models are statically assigned.
    """
    thread_id = str(uuid.uuid4())
    models = model_assignment or list(MODEL_POOL)

    yield _ev("task_started",
               task_id=task.task_id,
               prompt=task.prompt,
               family=task.family,
               config="A4",
               mode=mode.value,
               swarm_size=len(agent_ids),
               agent_ids=agent_ids,
               model_assignment=models)

    yield _ev("board_update", open=1, claimed=0, done=0,
               detail="Self-organising swarm (multi-model) — building LangGraph A4")

    try:
        compiled, _, _ = build_a4_graph(
            agent_ids=agent_ids,
            client=client,
            model_assignment=models,
            mode=mode,
        )

        initial_state = make_dashboard_state(
            task=task,
            agent_ids=agent_ids,
            run_id=thread_id,
            verification_mode=mode.value,
            config_name="A4",
        )

        yield from _stream_graph(compiled, initial_state, thread_id)

    except Exception as exc:
        yield _ev("error", message=str(exc), config="A4",
                   detail="LangGraph A4 execution failed")
        raise


# ── Public API ────────────────────────────────────────────────────────────────
__all__ = ["run_c1", "run_a1", "run_a2", "run_a3", "run_a4"]
