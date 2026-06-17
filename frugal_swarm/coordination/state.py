"""
LangGraph SwarmState schema.

All mutable swarm state lives here. LangGraph serialises it to SQLite via the
built-in SqliteSaver checkpointer so runs are pauseable and resumable.

DESIGN: every field that multiple nodes write to concurrently uses
  Annotated[list[T], operator.add]   →  LangGraph APPENDS, never overwrites
All other fields are last-write-wins (standard TypedDict behaviour).

DASHBOARD STREAMING:
  The `events` field is append-only. Every LangGraph node returns a list of
  dashboard event dicts under the "events" key.  The dashboard runner reads
  these from the stream() delta and forwards them as SSE to the browser.
  This means the browser sees events in real time as each node completes —
  without any polling or manual threading needed.
"""
from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional

from langgraph.graph import MessagesState


# ── Task record ────────────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id: str
    family: str
    prompt: str
    reference: str
    priority: int = 0
    status: str = "open"          # "open" | "claimed" | "done"
    claimed_by: Optional[str] = None
    result: Optional[str] = None
    tokens_used: int = 0
    latency_s: float = 0.0
    uncertainty: Optional[float] = None
    verified: bool = False
    verification_result: Optional[str] = None
    round_num: int = 0


# ── Agent contribution record ──────────────────────────────────────────────────

@dataclass
class AgentContribution:
    agent_id: str
    task_id: str
    response_text: str
    embedding: list[float]        # stored as list for JSON serialisability
    shapley_score: float = 0.0
    round_num: int = 0


# ── SwarmState ────────────────────────────────────────────────────────────────

class SwarmState(MessagesState):
    """
    Typed dict used as the LangGraph graph state for ALL configurations.

    Fields annotated with Annotated[list, operator.add] are APPEND-ONLY:
    LangGraph merges them automatically across nodes. All other fields
    are last-write-wins.

    DASHBOARD STREAMING
    -------------------
    `events`  — every node appends dashboard event dicts here.
                 The runner reads delta["events"] after each node and
                 forwards events directly to the SSE stream.

    CONFIG-SPECIFIC FIELDS
    ----------------------
    pipeline_responses  — A1/A2: accumulates each role's output in the pipeline
    current_task        — Dashboard single-task runs (not used in batch runner)
    final_response      — Set by the last node; read for final_output event
    """

    # ── Core task board (batch experiment runner) ──────────────────────────────
    task_queue:       list                            # list[Task], priority-sorted
    completed_tasks:  Annotated[list, operator.add]  # append-only completed tasks
    contributions:    Annotated[list, operator.add]  # append-only AgentContributions

    # ── DAG and role tracking ──────────────────────────────────────────────────
    dag:          dict                    # agent_id → list[agent_id] it informs
    task_history: dict                    # agent_id → list[family] (role monitor)

    # ── Experiment metadata ───────────────────────────────────────────────────
    run_id:            str
    swarm_size:        int
    verification_mode: str               # "full" | "selective" | "none"
    round_num:         int               # incremented by coordinator node
    config_name:       str               # "C1" | "A1" | "A2" | "A3" | "A4"
    seed:              int

    # ── Accumulated metrics ───────────────────────────────────────────────────
    total_tokens:           int
    total_tasks_attempted:  int
    total_tasks_succeeded:  int

    # ── Misc ──────────────────────────────────────────────────────────────────
    agent_ids:      list                 # list[str]
    stop_requested: bool

    # ══════════════════════════════════════════════════════════════════════════
    # DASHBOARD STREAMING — append-only event accumulator
    # Each LangGraph node returns {"events": [new_event, ...]}
    # The runner reads delta["events"] from stream() and forwards to SSE.
    # ══════════════════════════════════════════════════════════════════════════
    events: Annotated[list, operator.add]

    # ── A1/A2 fixed-role pipeline ────────────────────────────────────────────
    pipeline_responses: Annotated[list, operator.add]  # one str per role step

    # ── Single-task dashboard ─────────────────────────────────────────────────
    current_task: Any          # Task | None — set for single-task dashboard runs

    # ── Final answer (written by verifier / finaliser node) ──────────────────
    final_response: str

    # ── Collaborative decomposition (A1/A2/A3/A4) ───────────────────────────
    subtasks: list             # list[str] — set by planner/decomposer node


# ── Initial state helpers ─────────────────────────────────────────────────────

def make_initial_state(
    tasks: list[Task],
    agent_ids: list[str],
    run_id: str,
    verification_mode: str,
    config_name: str,
    seed: int,
) -> dict[str, Any]:
    """Build the initial SwarmState dict for a BATCH experiment run."""
    return {
        # Core
        "messages":             [],
        "task_queue":           sorted(tasks, key=lambda t: -t.priority),
        "completed_tasks":      [],
        "contributions":        [],
        "dag":                  {aid: [] for aid in agent_ids},
        "task_history":         {aid: [] for aid in agent_ids},
        # Metadata
        "run_id":               run_id,
        "swarm_size":           len(agent_ids),
        "verification_mode":    verification_mode,
        "round_num":            0,
        "config_name":          config_name,
        "seed":                 seed,
        # Metrics
        "total_tokens":         0,
        "total_tasks_attempted": 0,
        "total_tasks_succeeded": 0,
        # Misc
        "agent_ids":            agent_ids,
        "stop_requested":       False,
        # Dashboard streaming
        "events":               [],
        # Pipeline (A1/A2)
        "pipeline_responses":   [],
        # Single-task
        "current_task":         None,
        "final_response":       "",
        # Collaborative decomposition
        "subtasks":             [],
    }


def make_dashboard_state(
    task: Task,
    agent_ids: list[str],
    run_id: str,
    verification_mode: str,
    config_name: str,
) -> dict[str, Any]:
    """
    Build the initial SwarmState dict for a single-task DASHBOARD run.

    Uses `current_task` instead of a task queue — simpler for one-shot runs.
    """
    return {
        # Core
        "messages":             [],
        "task_queue":           [],           # unused in dashboard runs
        "completed_tasks":      [],
        "contributions":        [],
        "dag":                  {aid: [] for aid in agent_ids},
        "task_history":         {aid: [] for aid in agent_ids},
        # Metadata
        "run_id":               run_id,
        "swarm_size":           len(agent_ids),
        "verification_mode":    verification_mode,
        "round_num":            0,
        "config_name":          config_name,
        "seed":                 42,
        # Metrics
        "total_tokens":         0,
        "total_tasks_attempted": 1,
        "total_tasks_succeeded": 0,
        # Misc
        "agent_ids":            agent_ids,
        "stop_requested":       False,
        # Dashboard streaming
        "events":               [],
        # Pipeline (A1/A2)
        "pipeline_responses":   [],
        # Single-task
        "current_task":         task,
        "final_response":       "",
        # Collaborative decomposition
        "subtasks":             [],
    }


# ── Event helper (shared across all graph modules) ────────────────────────────

def make_event(event_type: str, **data) -> dict:
    """Create a dashboard event dict with a UTC timestamp."""
    return {"type": event_type, "ts": time.time(), **data}
