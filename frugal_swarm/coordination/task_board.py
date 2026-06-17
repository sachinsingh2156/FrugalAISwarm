"""
Shared Task Board.

Agents claim tasks with an atomic check-and-set on the LangGraph state.
No coordinator assigns tasks; agents self-select by priority order.
This is the observable behaviour that answers RQ1.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from frugal_swarm.coordination.state import Task, SwarmState


def view_open_tasks(state: dict[str, Any]) -> list[Task]:
    """Return tasks that are currently open (not claimed or done)."""
    return [t for t in state["task_queue"] if t.status == "open"]


def claim_task(state: dict[str, Any], agent_id: str) -> tuple[Task | None, dict[str, Any]]:
    """
    Atomically claim the highest-priority open task for agent_id.
    Returns (task, updated_state).  Returns (None, state) if no tasks are open.
    """
    queue = [deepcopy(t) for t in state["task_queue"]]
    for task in queue:
        if task.status == "open":
            task.status = "claimed"
            task.claimed_by = agent_id
            new_state = {**state, "task_queue": queue}
            return task, new_state
    return None, state


def submit_result(
    state: dict[str, Any],
    task: Task,
    result: str,
    tokens_used: int,
    latency_s: float,
    uncertainty: float | None = None,
    verified: bool = False,
    verification_result: str | None = None,
) -> dict[str, Any]:
    """
    Mark a task as done, update its result fields, and move it to the
    completed list.  Updates total_tokens and task_history for the agent.
    """
    queue = [deepcopy(t) for t in state["task_queue"]]
    completed = list(state.get("completed_tasks", []))
    task_history = deepcopy(state.get("task_history", {}))

    for t in queue:
        if t.task_id == task.task_id:
            t.status = "done"
            t.result = result
            t.tokens_used = tokens_used
            t.latency_s = latency_s
            t.uncertainty = uncertainty
            t.verified = verified
            t.verification_result = verification_result
            t.round_num = state.get("round_num", 0)
            completed.append(t)

            # Update task_history for role-stability monitor
            agent_id = task.claimed_by or ""
            if agent_id:
                hist = task_history.get(agent_id, [])
                hist.append(t.family)
                task_history[agent_id] = hist
            break

    new_total_tokens = state.get("total_tokens", 0) + tokens_used
    new_attempted = state.get("total_tasks_attempted", 0) + 1

    return {
        **state,
        "task_queue": queue,
        "completed_tasks": completed,
        "task_history": task_history,
        "total_tokens": new_total_tokens,
        "total_tasks_attempted": new_attempted,
    }


def all_tasks_done(state: dict[str, Any]) -> bool:
    """Return True when every task in the queue is done."""
    return all(t.status == "done" for t in state["task_queue"])


def board_summary(state: dict[str, Any]) -> dict[str, int]:
    """Quick status count for logging."""
    counts: dict[str, int] = {"open": 0, "claimed": 0, "done": 0}
    for t in state["task_queue"]:
        counts[t.status] = counts.get(t.status, 0) + 1
    return counts
