"""
LangGraph StateGraph builders — collaborative multi-agent architecture.

DESIGN PRINCIPLE
----------------
Every configuration implements REAL agent collaboration — tasks are decomposed
into subtasks, each agent owns one piece, and results are synthesised into a
final answer.  No agent re-does the whole task independently.

A1 / A2  (Fixed-Role):
    Planner  → Executor  → Verifier → END
    • Planner:  breaks task into 3 numbered subtasks
    • Executor: completes every subtask and formats the answers
    • Verifier: synthesises the executor's work into the polished final answer

A3 / A4  (Self-Organising):
    Decomposer → worker_0 → worker_1 → worker_2 → Aggregator → END
    • Decomposer (agent_0): splits task into N subtasks (one per agent)
    • Each worker: executes only its assigned subtask
    • Aggregator:  Shapley-scored synthesis of all workers into the final answer

C1  (Baseline):
    Single agent → END
    Direct response, no decomposition.

DASHBOARD STREAMING
-------------------
Every node appends event dicts to {"events": [...]}.
graph.stream(stream_mode="updates") yields per-node deltas which the
dashboard runner forwards as SSE.  New event types:
  task_decomposed     — subtasks list visible in dashboard
  subtask_assigned    — which agent got which subtask
  agent_response      — agent's result (includes subtask_num for workers)
  aggregation_complete— aggregator summary with all contributions
"""
from __future__ import annotations

import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import (
    LANGGRAPH_DB_PATH, MAX_ROUNDS, VerificationMode,
)
from frugal_swarm.coordination.state import (
    SwarmState, AgentContribution, make_event,
)
from frugal_swarm.coordination.agent_node import AGENT_SYSTEM_PROMPT
from frugal_swarm.coordination.shapley import ShapleyEstimator
from frugal_swarm.coordination.role_monitor import RoleMonitor
from frugal_swarm.model.ollama_client import OllamaClient
from frugal_swarm.model.embedder import embed
from frugal_swarm.corpus.rubrics import score as rubric_score

_checkpointer: SqliteSaver | None = None
_checkpointer_conn: sqlite3.Connection | None = None


def _make_checkpointer() -> SqliteSaver:
    global _checkpointer, _checkpointer_conn
    if _checkpointer is None:
        db_path = Path(LANGGRAPH_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _checkpointer_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _checkpointer = SqliteSaver(_checkpointer_conn)
    return _checkpointer


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

_PLANNER_SYS = (
    "You are the Planner in a collaborative educational AI team. "
    "Your ONLY job is to decompose the given task into exactly 3 clear, numbered subtasks. "
    "Do NOT produce any content or answers yet — only the plan. "
    "Be specific: each subtask must be self-contained and actionable."
)

_EXECUTOR_SYS = (
    "You are the Executor in a collaborative educational AI team. "
    "You receive a numbered plan from the Planner and must complete every subtask fully. "
    "Provide accurate, educationally appropriate content for each numbered item."
)

_VERIFIER_SYS = (
    "You are the Pedagogical Verifier in a collaborative educational AI team. "
    "You receive the Planner's breakdown and the Executor's work. "
    "Your job: synthesise all subtask results into one coherent, polished final answer. "
    "Correct errors, ensure clarity and inclusivity, and match the stated learning objectives. "
    "Output the final complete answer — nothing else."
)

_DECOMPOSER_SYS = (
    "You are the Task Decomposer in a collaborative AI swarm. "
    "Break the given task into exactly {n} independent subtasks — one per team member. "
    "Each subtask must be a meaningful, self-contained piece of the overall work. "
    "Use exactly the format: SUBTASK 1: ... / SUBTASK 2: ... / SUBTASK 3: ..."
)

_WORKER_SYS = (
    "You are {agent_id} in a collaborative AI swarm. "
    "You have been assigned one specific subtask. Complete it thoroughly and accurately. "
    "Your result will be combined with your teammates' work to form the final answer."
)

_AGGREGATOR_SYS = (
    "You are the Aggregator in a collaborative AI swarm. "
    "You receive results from all team members. "
    "Synthesise them into one complete, coherent, well-structured final answer. "
    "Ensure all subtasks are addressed and the response reads naturally."
)


# ─────────────────────────────────────────────────────────────────────────────
# Text parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_subtasks(text: str, n: int = 3) -> list[str]:
    """Parse SUBTASK N: lines; fall back to numbered list; fill with placeholders."""
    subtasks: list[str] = []

    # Try "SUBTASK N:" pattern (case-insensitive)
    for i in range(1, n + 1):
        pat = re.compile(rf"SUBTASK\s+{i}\s*:", re.IGNORECASE)
        m = pat.search(text)
        if m:
            rest = text[m.end():].strip()
            # Take up to the next SUBTASK marker or end
            nxt = re.search(rf"SUBTASK\s+{i+1}\s*:", rest, re.IGNORECASE)
            chunk = rest[:nxt.start()].strip() if nxt else rest.strip()
            # Keep only first line if multi-line
            line = chunk.splitlines()[0].strip() if chunk else ""
            subtasks.append(line)

    if len(subtasks) == n:
        return subtasks

    # Fall back: numbered list patterns "1. " / "1) "
    subtasks = []
    matches = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s+(.+)", text)
    if matches:
        subtasks = [m.strip() for m in matches[:n]]

    # Last resort: split into n chunks
    if not subtasks:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        subtasks = lines[:n] if lines else []

    while len(subtasks) < n:
        subtasks.append(f"Part {len(subtasks) + 1} of the overall task")

    return subtasks[:n]


def _extract_subtask_responses(text: str, n: int = 3) -> list[str]:
    """Parse SUBTASK N RESPONSE: sections; fall back gracefully."""
    sections: dict[int, list[str]] = {}
    current: int | None = None

    for line in text.splitlines():
        s = line.strip()
        matched = False
        for i in range(1, n + 1):
            if re.match(rf"SUBTASK\s+{i}\s+(RESPONSE\s*)?:", s, re.IGNORECASE):
                current = i
                rest = s[s.index(":") + 1:].strip()
                sections[i] = [rest] if rest else []
                matched = True
                break
        if not matched and current is not None:
            sections[current].append(line)

    results = ["\n".join(sections.get(i, [])).strip() for i in range(1, n + 1)]

    if not any(results):
        return [text] + [""] * (n - 1)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# C1 — Single-Agent Baseline
# ─────────────────────────────────────────────────────────────────────────────

def _make_c1_agent_fn(agent_id: str, client: OllamaClient, max_tokens: int):
    def node_fn(state: dict) -> dict:
        task = state["current_task"]
        events: list[dict] = []

        events.append(make_event("agent_thinking", agent_id=agent_id,
                                 detail="Single agent is processing the task..."))

        t0 = time.perf_counter()
        resp = client.generate(prompt=task.prompt, system=AGENT_SYSTEM_PROMPT,
                               temperature=0.7, max_tokens=max_tokens)
        latency = time.perf_counter() - t0
        text = resp["text"].strip()
        tokens = resp["tokens_used"]
        total_tokens = state.get("total_tokens", 0) + tokens
        success = rubric_score(task.family, text, task.reference)

        events.append(make_event("agent_response", agent_id=agent_id,
                                 response=text, tokens=tokens, latency_s=round(latency, 2)))
        events.append(make_event("final_output", response=text, total_tokens=total_tokens,
                                 verified=False, best_agent=agent_id,
                                 reference=task.reference, detail="Single-agent result"))
        events.append(make_event("run_complete", total_tokens=total_tokens,
                                 success_score=round(success, 3), passed=success >= 0.4,
                                 config="C1", mode="none", swarm_size=1, rounds=1))
        return {
            "final_response": text,
            "total_tokens": total_tokens,
            "total_tasks_succeeded": 1 if success >= 0.4 else 0,
            "events": events,
        }
    node_fn.__name__ = agent_id
    return node_fn


def build_c1_graph(agent_ids: list[str], client: OllamaClient, max_tokens: int = 400):
    """C1: single-agent direct response."""
    graph = StateGraph(SwarmState)
    graph.add_node("agent", _make_c1_agent_fn(agent_ids[0], client, max_tokens))
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile(checkpointer=_make_checkpointer())
    # Return same tuple shape as build_c3_graph for runner compatibility
    return compiled, ShapleyEstimator(agent_ids=agent_ids), RoleMonitor(agent_ids=agent_ids)


# ─────────────────────────────────────────────────────────────────────────────
# A1 / A2 — Fixed-Role Pipeline:  Planner → Executor → Verifier
# ─────────────────────────────────────────────────────────────────────────────

def _make_planner_node(agent_id: str, client: OllamaClient, config_name: str,
                       max_tokens: int, model_override: str | None = None):
    def node_fn(state: dict) -> dict:
        task = state["current_task"]
        events: list[dict] = []

        events.append(make_event("agent_thinking", agent_id=agent_id, role="Planner",
                                 detail=f"Planner ({agent_id}) is analysing the task and creating a plan..."))

        prompt = (
            f"Task:\n{task.prompt}\n\n"
            "Decompose this into exactly 3 subtasks. Use this exact format:\n"
            "SUBTASK 1: [first subtask — be specific]\n"
            "SUBTASK 2: [second subtask — be specific]\n"
            "SUBTASK 3: [third subtask — be specific]"
        )
        t0 = time.perf_counter()
        resp = client.generate(prompt=prompt, system=_PLANNER_SYS, temperature=0.5,
                               max_tokens=max_tokens, model_override=model_override)
        latency = time.perf_counter() - t0
        text = resp["text"].strip()
        tokens = resp["tokens_used"]
        subtasks = _extract_subtasks(text, n=3)

        events.append(make_event("task_decomposed",
                                 agent_id=agent_id, role="Planner",
                                 config=config_name,
                                 plan_text=text, subtasks=subtasks,
                                 tokens=tokens, latency_s=round(latency, 2),
                                 detail=f"Plan: {len(subtasks)} subtasks created"))
        events.append(make_event("agent_response",
                                 agent_id=agent_id, role="Planner",
                                 response=text, tokens=tokens,
                                 latency_s=round(latency, 2),
                                 subtasks=subtasks,
                                 detail=f"Planner created {len(subtasks)} subtasks"))

        return {
            "pipeline_responses": [text],
            "subtasks": subtasks,
            "total_tokens": state.get("total_tokens", 0) + tokens,
            "events": events,
        }
    node_fn.__name__ = f"planner_{agent_id}"
    return node_fn


def _make_executor_node(agent_id: str, client: OllamaClient,
                        max_tokens: int, model_override: str | None = None):
    def node_fn(state: dict) -> dict:
        task = state["current_task"]
        pipeline = list(state.get("pipeline_responses") or [])
        subtasks = list(state.get("subtasks") or [])
        plan_text = pipeline[0] if pipeline else ""
        events: list[dict] = []

        events.append(make_event("agent_thinking", agent_id=agent_id, role="Executor",
                                 detail=f"Executor ({agent_id}) is working through each subtask..."))

        subtask_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(subtasks)) if subtasks else plan_text
        prompt = (
            f"Original task:\n{task.prompt}\n\n"
            f"Subtasks to complete:\n{subtask_block}\n\n"
            "Complete ALL subtasks. Use this format:\n"
            "SUBTASK 1 RESPONSE:\n[your answer]\n\n"
            "SUBTASK 2 RESPONSE:\n[your answer]\n\n"
            "SUBTASK 3 RESPONSE:\n[your answer]"
        )
        t0 = time.perf_counter()
        resp = client.generate(prompt=prompt, system=_EXECUTOR_SYS, temperature=0.7,
                               max_tokens=max_tokens * 2, model_override=model_override)
        latency = time.perf_counter() - t0
        text = resp["text"].strip()
        tokens = resp["tokens_used"]
        subtask_responses = _extract_subtask_responses(text, n=len(subtasks) or 3)

        events.append(make_event("agent_response",
                                 agent_id=agent_id, role="Executor",
                                 response=text, tokens=tokens, latency_s=round(latency, 2),
                                 subtask_responses=subtask_responses,
                                 subtasks=subtasks,
                                 detail=f"Executor completed {len([r for r in subtask_responses if r])} subtasks"))

        return {
            "pipeline_responses": [text],
            "total_tokens": state.get("total_tokens", 0) + tokens,
            "events": events,
        }
    node_fn.__name__ = f"executor_{agent_id}"
    return node_fn


def _make_verifier_node(agent_id: str, client: OllamaClient, config_name: str,
                        max_tokens: int, model_override: str | None = None):
    def node_fn(state: dict) -> dict:
        task = state["current_task"]
        pipeline = list(state.get("pipeline_responses") or [])
        plan_text = pipeline[0] if len(pipeline) > 0 else ""
        exec_text = pipeline[1] if len(pipeline) > 1 else ""
        subtasks = list(state.get("subtasks") or [])
        events: list[dict] = []

        events.append(make_event("agent_thinking", agent_id=agent_id, role="Verifier",
                                 detail=f"Verifier ({agent_id}) is synthesising all work into the final answer..."))

        prompt = (
            f"Original task:\n{task.prompt}\n\n"
            f"Planner's breakdown:\n{plan_text}\n\n"
            f"Executor's work:\n{exec_text}\n\n"
            "Synthesise all subtask results into ONE complete, coherent answer. "
            "Fix any errors. Ensure it is educationally sound."
        )
        t0 = time.perf_counter()
        resp = client.generate(prompt=prompt, system=_VERIFIER_SYS, temperature=0.3,
                               max_tokens=max_tokens, model_override=model_override)
        latency = time.perf_counter() - t0
        text = resp["text"].strip()
        tokens = resp["tokens_used"]
        total_tokens = state.get("total_tokens", 0) + tokens
        success = rubric_score(task.family, text, task.reference)

        events.append(make_event("agent_response",
                                 agent_id=agent_id, role="Verifier",
                                 response=text, tokens=tokens, latency_s=round(latency, 2),
                                 detail="Verifier synthesis complete"))
        events.append(make_event("pipeline_summary",
                                 config=config_name,
                                 roles=["Planner", "Executor", "Verifier"],
                                 subtasks=subtasks,
                                 plan=plan_text, execution=exec_text, final=text,
                                 detail="Full pipeline complete"))
        events.append(make_event("final_output",
                                 response=text, total_tokens=total_tokens,
                                 verified=True, best_agent=agent_id,
                                 reference=task.reference,
                                 detail=f"Verified synthesis by {agent_id} ({config_name})"))
        events.append(make_event("run_complete",
                                 total_tokens=total_tokens,
                                 success_score=round(success, 3), passed=success >= 0.4,
                                 config=config_name, mode="none", swarm_size=3, rounds=3))
        return {
            "pipeline_responses": [text],
            "final_response": text,
            "total_tokens": total_tokens,
            "total_tasks_succeeded": 1 if success >= 0.4 else 0,
            "events": events,
        }
    node_fn.__name__ = f"verifier_{agent_id}"
    return node_fn


def _build_collaborative_pipeline(
    agent_ids: list[str],
    client: OllamaClient,
    config_name: str,
    max_tokens: int = 500,
    model_assignment: list[str] | None = None,
):
    """Build Planner → Executor → Verifier graph (A1 or A2)."""
    role_agents = agent_ids[:3]
    models = model_assignment[:3] if model_assignment else [None, None, None]

    graph = StateGraph(SwarmState)
    graph.add_node("planner",
                   _make_planner_node(role_agents[0], client, config_name, max_tokens, models[0]))
    graph.add_node("executor",
                   _make_executor_node(role_agents[1], client, max_tokens, models[1]))
    graph.add_node("verifier",
                   _make_verifier_node(role_agents[2], client, config_name, max_tokens, models[2]))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "verifier")
    graph.add_edge("verifier", END)

    return graph.compile(checkpointer=_make_checkpointer())


def build_c2_edu_graph(agent_ids: list[str], client: OllamaClient, max_tokens: int = 500):
    """A1: Planner → Executor → Verifier (single shared model)."""
    return _build_collaborative_pipeline(agent_ids, client, "A1", max_tokens)


def build_a2_graph(agent_ids: list[str], client: OllamaClient,
                   model_assignment: list[str], max_tokens: int = 500):
    """A2: Planner → Executor → Verifier (per-role model: qwen → gemma → phi3)."""
    return _build_collaborative_pipeline(agent_ids, client, "A2", max_tokens, model_assignment)


# ─────────────────────────────────────────────────────────────────────────────
# A3 / A4 — Self-Organising:  Decomposer → Workers → Aggregator
# ─────────────────────────────────────────────────────────────────────────────

def _make_decomposer_fn(agent_id: str, all_agent_ids: list[str], client: OllamaClient,
                        max_tokens: int, model_override: str | None = None):
    n = len(all_agent_ids)

    def node_fn(state: dict) -> dict:
        task = state["current_task"]
        events: list[dict] = []

        events.append(make_event("agent_thinking", agent_id=agent_id,
                                 role="Decomposer",
                                 detail=f"Decomposer ({agent_id}) is splitting the task for {n} agents..."))

        fmt = "\n".join(f"SUBTASK {i+1}: [subtask {i+1} description]" for i in range(n))
        prompt = (
            f"Task:\n{task.prompt}\n\n"
            f"Split this into exactly {n} independent subtasks (one per agent). "
            f"Each must be a meaningful self-contained piece.\n\nFormat:\n{fmt}"
        )
        t0 = time.perf_counter()
        resp = client.generate(prompt=prompt,
                               system=_DECOMPOSER_SYS.format(n=n),
                               temperature=0.5, max_tokens=max_tokens,
                               model_override=model_override)
        latency = time.perf_counter() - t0
        text = resp["text"].strip()
        tokens = resp["tokens_used"]
        subtasks = _extract_subtasks(text, n=n)

        assignment_events = []
        for i, (aid, st) in enumerate(zip(all_agent_ids, subtasks)):
            assignment_events.append(
                make_event("subtask_assigned",
                           agent_id=aid, subtask_num=i + 1, subtask_text=st,
                           detail=f"{aid} → Subtask {i+1}: {st[:70]}{'...' if len(st) > 70 else ''}"))

        events.append(make_event("task_decomposed",
                                 agent_id=agent_id, n_agents=n,
                                 plan_text=text, subtasks=subtasks,
                                 tokens=tokens, latency_s=round(latency, 2),
                                 detail=f"Task split into {n} subtasks"))
        events.append(make_event("agent_response",
                                 agent_id=agent_id, role="Decomposer",
                                 response=text, tokens=tokens,
                                 latency_s=round(latency, 2),
                                 subtasks=subtasks,
                                 detail=f"Decomposer split task into {n} subtasks"))
        events.extend(assignment_events)

        return {
            "pipeline_responses": [text],
            "subtasks": subtasks,
            "total_tokens": state.get("total_tokens", 0) + tokens,
            "events": events,
        }
    node_fn.__name__ = f"decomposer_{agent_id}"
    return node_fn


def _make_worker_fn(worker_agent_id: str, worker_idx: int, all_agent_ids: list[str],
                    client: OllamaClient, shapley_est: ShapleyEstimator,
                    role_mon: RoleMonitor, max_tokens: int,
                    model_override: str | None = None):
    def node_fn(state: dict) -> dict:
        task = state["current_task"]
        subtasks = list(state.get("subtasks") or [])
        my_subtask = subtasks[worker_idx] if worker_idx < len(subtasks) else task.prompt
        events: list[dict] = []

        events.append(make_event("agent_thinking",
                                 agent_id=worker_agent_id,
                                 subtask_num=worker_idx + 1,
                                 subtask_text=my_subtask,
                                 detail=f"{worker_agent_id} working on subtask {worker_idx+1}: {my_subtask[:60]}..."))

        prompt = (
            f"Overall task context:\n{task.prompt}\n\n"
            f"Your assigned subtask:\n{my_subtask}\n\n"
            "Complete your subtask thoroughly. Your result will be combined with teammates'."
        )
        t0 = time.perf_counter()
        resp = client.generate(prompt=prompt,
                               system=_WORKER_SYS.format(agent_id=worker_agent_id),
                               temperature=0.7, max_tokens=max_tokens,
                               model_override=model_override)
        latency = time.perf_counter() - t0
        text = resp["text"].strip()
        tokens = resp["tokens_used"]

        emb = embed(text).tolist()
        contribution = AgentContribution(
            agent_id=worker_agent_id,
            task_id=task.task_id,
            response_text=text,
            embedding=emb,
            round_num=0,
        )
        role_mon.update(worker_agent_id, task.family, np.array(emb))

        events.append(make_event("agent_response",
                                 agent_id=worker_agent_id,
                                 subtask_num=worker_idx + 1,
                                 subtask_text=my_subtask,
                                 response=text, tokens=tokens,
                                 latency_s=round(latency, 2),
                                 detail=f"{worker_agent_id} completed subtask {worker_idx+1}"))
        return {
            "pipeline_responses": [text],
            "contributions": [contribution],
            "total_tokens": state.get("total_tokens", 0) + tokens,
            "events": events,
        }
    node_fn.__name__ = f"worker_{worker_agent_id}"
    return node_fn


def _make_aggregator_fn(all_agent_ids: list[str], shapley_est: ShapleyEstimator,
                        client: OllamaClient, config_name: str,
                        mode: VerificationMode, max_tokens: int,
                        model_assignment: list[str] | None = None):
    n = len(all_agent_ids)

    def node_fn(state: dict) -> dict:
        task = state["current_task"]
        pipeline = list(state.get("pipeline_responses") or [])
        subtasks = list(state.get("subtasks") or [])
        contributions = list(state.get("contributions") or [])
        total_tokens = state.get("total_tokens", 0)

        # pipeline[0] = decomposer plan; pipeline[1..n] = worker results
        plan_text = pipeline[0] if pipeline else ""
        worker_results = pipeline[1: n + 1] if len(pipeline) > 1 else []

        events: list[dict] = []
        events.append(make_event("agent_thinking", agent_id="aggregator",
                                 detail="Aggregator is synthesising all agent results..."))

        # Shapley scores from worker embeddings
        best_agent = all_agent_ids[0]
        shapley_scores: dict[str, float] = {}
        if contributions:
            embeddings = {c.agent_id: np.array(c.embedding) for c in contributions}
            try:
                shapley_est.update(embeddings)
                shapley_scores = {k: round(v, 4) for k, v in shapley_est.rolling_scores().items()}
                best_agent = max(shapley_scores, key=lambda k: shapley_scores[k])
                events.append(make_event("shapley_update",
                                         scores=shapley_scores, best_agent=best_agent, round=1))
            except Exception:
                pass

        # Choose aggregator model = model of best-scoring agent (A4) or shared (A3)
        agg_model: str | None = None
        if model_assignment:
            try:
                best_idx = all_agent_ids.index(best_agent)
                agg_model = model_assignment[best_idx] if best_idx < len(model_assignment) else None
            except ValueError:
                pass

        # Build synthesis prompt
        worker_section = "\n\n".join(
            f"--- {all_agent_ids[i]} | Subtask {i+1}"
            f"{': ' + subtasks[i] if i < len(subtasks) else ''} ---\n{r}"
            for i, r in enumerate(worker_results)
        )
        prompt = (
            f"Original task:\n{task.prompt}\n\n"
            f"Decomposition plan:\n{plan_text}\n\n"
            f"Agent contributions:\n{worker_section}\n\n"
            "Synthesise ALL contributions into ONE complete, coherent final answer. "
            "Ensure every subtask is addressed and the response flows naturally."
        )
        t0 = time.perf_counter()
        resp = client.generate(prompt=prompt, system=_AGGREGATOR_SYS,
                               temperature=0.4, max_tokens=max_tokens,
                               model_override=agg_model)
        latency = time.perf_counter() - t0
        agg_text = resp["text"].strip()
        agg_tokens = resp["tokens_used"]
        total_tokens += agg_tokens

        # Optional verification
        verified = False
        if mode != VerificationMode.NONE and agg_text:
            try:
                from frugal_swarm.verification.modes import make_verifier
                vfn = make_verifier(mode, client)
                verified, v_result = vfn(task, agg_text)
                if v_result and verified:
                    agg_text = v_result
                events.append(make_event("verification_result",
                                         verified=verified, mode=mode.value))
            except Exception:
                pass

        success = rubric_score(task.family, agg_text, task.reference)

        events.append(make_event("aggregation_complete",
                                 agent_results=worker_results,
                                 subtasks=subtasks,
                                 agent_ids=all_agent_ids,
                                 shapley_scores=shapley_scores,
                                 best_agent=best_agent,
                                 response=agg_text,
                                 tokens=agg_tokens, latency_s=round(latency, 2),
                                 detail=f"Aggregated {len(worker_results)} contributions (best: {best_agent})"))
        events.append(make_event("final_output",
                                 response=agg_text, total_tokens=total_tokens,
                                 verified=verified, best_agent=best_agent,
                                 reference=task.reference,
                                 detail=f"Aggregated answer ({config_name})"))
        events.append(make_event("run_complete",
                                 total_tokens=total_tokens,
                                 success_score=round(success, 3), passed=success >= 0.4,
                                 config=config_name, mode=mode.value,
                                 swarm_size=n, rounds=1))
        return {
            "final_response": agg_text,
            "total_tokens": total_tokens,
            "total_tasks_succeeded": 1 if success >= 0.4 else 0,
            "events": events,
        }
    return node_fn


def _build_swarm_graph(
    agent_ids: list[str],
    client: OllamaClient,
    config_name: str,
    mode: VerificationMode,
    max_tokens: int,
    role_mon: RoleMonitor | None,
    model_assignment: list[str] | None = None,
):
    """Generic builder for A3/A4: Decomposer → Workers → Aggregator."""
    shapley_est = ShapleyEstimator(agent_ids=agent_ids)
    if role_mon is None:
        role_mon = RoleMonitor(agent_ids=agent_ids)
    models = model_assignment or [None] * len(agent_ids)

    graph = StateGraph(SwarmState)

    # Decomposer (always uses agent_0's model)
    graph.add_node("decomposer",
                   _make_decomposer_fn(agent_ids[0], agent_ids, client,
                                       max_tokens, model_override=models[0]))

    # One worker per agent
    for i, aid in enumerate(agent_ids):
        graph.add_node(f"worker_{aid}",
                       _make_worker_fn(aid, i, agent_ids, client,
                                       shapley_est, role_mon, max_tokens,
                                       model_override=models[i]))

    # Aggregator
    graph.add_node("aggregator",
                   _make_aggregator_fn(agent_ids, shapley_est, client,
                                       config_name, mode, max_tokens,
                                       model_assignment=model_assignment))

    # Wire: decomposer → worker_0 → worker_1 → ... → aggregator → END
    graph.set_entry_point("decomposer")
    graph.add_edge("decomposer", f"worker_{agent_ids[0]}")
    for i in range(len(agent_ids) - 1):
        graph.add_edge(f"worker_{agent_ids[i]}", f"worker_{agent_ids[i + 1]}")
    graph.add_edge(f"worker_{agent_ids[-1]}", "aggregator")
    graph.add_edge("aggregator", END)

    compiled = graph.compile(checkpointer=_make_checkpointer())
    return compiled, shapley_est, role_mon


def build_c3_graph(
    agent_ids: list[str],
    client: OllamaClient,
    mode: VerificationMode = VerificationMode.NONE,
    temperature: float = 0.7,
    max_tokens: int = 400,
    use_round_robin_dag: bool = False,
    role_mon: RoleMonitor | None = None,
    max_rounds: int | None = None,
):
    """
    A3: self-organising swarm, single shared model.
    If N=1 (C1 baseline), delegates to build_c1_graph.
    Otherwise: Decomposer → Workers → Aggregator.
    """
    if len(agent_ids) == 1:
        compiled, shapley_est, rm = build_c1_graph(agent_ids, client, max_tokens)
        return compiled, shapley_est, rm

    return _build_swarm_graph(
        agent_ids=agent_ids, client=client, config_name="A3",
        mode=mode, max_tokens=max_tokens, role_mon=role_mon,
    )


def build_a4_graph(
    agent_ids: list[str],
    client: OllamaClient,
    model_assignment: list[str],
    mode: VerificationMode = VerificationMode.NONE,
    temperature: float = 0.7,
    max_tokens: int = 400,
    use_round_robin_dag: bool = False,
    role_mon: RoleMonitor | None = None,
    max_rounds: int | None = None,
):
    """
    A4: self-organising swarm, per-agent model.
    Decomposer → Workers (per-model) → Aggregator (best-agent model).
    """
    return _build_swarm_graph(
        agent_ids=agent_ids, client=client, config_name="A4",
        mode=mode, max_tokens=max_tokens, role_mon=role_mon,
        model_assignment=model_assignment,
    )
