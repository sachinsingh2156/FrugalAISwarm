"""
Response-conditioned DAG builder.

Each round, a directed acyclic graph is rebuilt from the latest rolling
Shapley scores.  Information flows from high-contributing agents to
lower-contributing agents (exactly as in SELFORG, Tastan et al. 2026).

The DAG is materialised as an adjacency list stored in SwarmState["dag"].
It influences the next round's prompts: each agent sees the outputs of agents
that have edges pointing to it (its "upstream" agents).
"""
from __future__ import annotations

from typing import Any


def build_dag(
    rolling_scores: dict[str, float],
    agent_ids: list[str],
) -> dict[str, list[str]]:
    """
    Build a DAG where high-scoring agents send their outputs to low-scoring agents.

    Strategy: sort agents by score descending.  The top-half agents are
    "sources"; the bottom-half are "sinks".  Each source is connected to
    each sink.  This is a simple but effective topology that matches SELFORG's
    intent without introducing cycles.

    Returns: adjacency list {agent_id: [agents it sends info TO]}
    """
    if len(agent_ids) <= 1:
        return {aid: [] for aid in agent_ids}

    # Sort agents by score, highest first
    ranked = sorted(agent_ids, key=lambda aid: rolling_scores.get(aid, 0), reverse=True)
    mid = max(1, len(ranked) // 2)
    sources = ranked[:mid]
    sinks = ranked[mid:]

    dag: dict[str, list[str]] = {aid: [] for aid in agent_ids}
    for src in sources:
        for sink in sinks:
            if src != sink:
                dag[src].append(sink)

    return dag


def get_upstream_outputs(
    agent_id: str,
    dag: dict[str, list[str]],
    last_round_outputs: dict[str, str],
) -> list[str]:
    """
    Return the outputs of agents that have a directed edge TO agent_id.
    (i.e., who sends information to this agent — the reverse of the adjacency list.)
    """
    upstream = [
        src for src, targets in dag.items()
        if agent_id in targets and src != agent_id
    ]
    return [last_round_outputs[src] for src in upstream if src in last_round_outputs]


def round_robin_dag(agent_ids: list[str]) -> dict[str, list[str]]:
    """
    Fallback static DAG (M2 fallback): each agent sends its output to the
    next agent in a ring.
    """
    n = len(agent_ids)
    return {agent_ids[i]: [agent_ids[(i + 1) % n]] for i in range(n)}
