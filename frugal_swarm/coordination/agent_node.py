"""
Identical Agent Node.

Each agent in the swarm runs this exact logic with NO role labels.
The system prompt is identical across all agents — this is the necessary
condition for RQ1 (emergent specialisation) to be testable.

Cycle per round:
  1. Claim the highest-priority open task from the shared task board.
  2. Optionally incorporate upstream context from the current DAG.
  3. Call Ollama to generate a response.
  4. Embed the response with all-MiniLM-L6-v2.
  5. Submit the result back to the task board.
  6. Log the contribution (embedding + Shapley score) for the round.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np

from frugal_swarm.coordination.state import AgentContribution, Task
from frugal_swarm.coordination.task_board import claim_task, submit_result
from frugal_swarm.coordination.dag_builder import get_upstream_outputs
from frugal_swarm.model.ollama_client import OllamaClient
from frugal_swarm.model.embedder import embed

# ── System prompt (identical for ALL agents — no role labels) ─────────────────
AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant working as part of a collaborative team.
Your job is to solve the given task as accurately and concisely as possible.
If you have access to insights from other team members, use them to improve your answer,
but do not simply repeat what they said — critically evaluate and synthesise.
Provide a clear, direct answer. Do not pad your response unnecessarily."""


def build_agent_prompt(
    task: Task,
    upstream_outputs: list[str],
) -> str:
    """
    Construct the agent's prompt for this task, optionally incorporating
    upstream context from the DAG.
    """
    lines = [f"Task: {task.prompt}"]

    if upstream_outputs:
        lines.append("\n--- Insights from team members (evaluate critically) ---")
        for i, out in enumerate(upstream_outputs, 1):
            lines.append(f"[Team member {i}]: {out.strip()}")
        lines.append("--- End of team insights ---\n")
        lines.append("Now provide YOUR answer, synthesising the above where useful:")
    else:
        lines.append("\nProvide your answer:")

    return "\n".join(lines)


class AgentNode:
    """
    Encapsulates one swarm agent.

    agent_id: unique string identifier (e.g., "agent_0")
    client:   shared OllamaClient (all agents use the same endpoint)
    verification_fn: optional callable(task, response) → (verified, result_str)
    """

    def __init__(
        self,
        agent_id: str,
        client: OllamaClient,
        verification_fn: Any | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        self.agent_id = agent_id
        self.client = client
        self.verification_fn = verification_fn
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(
        self,
        state: dict[str, Any],
        logger: Any | None = None,
    ) -> dict[str, Any]:
        """
        Execute one agent turn:  claim → infer → verify (optional) → submit.

        Returns a partial state update dict.
        """
        # ── 1. Claim a task ──────────────────────────────────────────────────
        task, state = claim_task(state, self.agent_id)
        if task is None:
            return state  # nothing to do this round

        # ── 2. Gather upstream context from DAG ─────────────────────────────
        dag = state.get("dag", {})
        last_round_outputs: dict[str, str] = {
            c.agent_id: c.response_text
            for c in state.get("contributions", [])
            if c.round_num == state.get("round_num", 0) - 1
        }
        upstream = get_upstream_outputs(self.agent_id, dag, last_round_outputs)

        # ── 3. Build prompt and call model ───────────────────────────────────
        prompt = build_agent_prompt(task, upstream)
        t0 = time.perf_counter()
        resp = self.client.generate(
            prompt=prompt,
            system=AGENT_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=True,  # needed for uncertainty signal
        )
        latency_s = time.perf_counter() - t0
        response_text: str = resp["text"].strip()
        tokens_used: int = resp["tokens_used"]
        mean_lp: float | None = resp.get("mean_log_prob")

        # ── 4. Compute uncertainty signal ────────────────────────────────────
        # Uncertainty = -mean_log_prob (higher → more uncertain)
        # None if Ollama doesn't return logprobs (version dependent)
        uncertainty: float | None = (
            -mean_lp if mean_lp is not None else None
        )

        # ── 5. Embed the response ────────────────────────────────────────────
        response_embedding: np.ndarray = embed(response_text)
        embedding_list: list[float] = response_embedding.tolist()

        # ── 6. Optional verification ─────────────────────────────────────────
        verified = False
        verification_result: str | None = None
        if self.verification_fn is not None:
            verified, verification_result = self.verification_fn(task, response_text)

        # ── 7. Submit result to task board ───────────────────────────────────
        state = submit_result(
            state=state,
            task=task,
            result=response_text,
            tokens_used=tokens_used,
            latency_s=latency_s,
            uncertainty=uncertainty,
            verified=verified,
            verification_result=verification_result,
        )

        # ── 8. Record contribution for this round ────────────────────────────
        contribution = AgentContribution(
            agent_id=self.agent_id,
            task_id=task.task_id,
            response_text=response_text,
            embedding=embedding_list,
            shapley_score=0.0,       # computed after all agents complete a round
            round_num=state.get("round_num", 0),
        )
        contributions = list(state.get("contributions", []))
        contributions.append(contribution)
        state = {**state, "contributions": contributions}

        # ── 9. Log trace ─────────────────────────────────────────────────────
        if logger is not None:
            logger.log_agent_action(
                run_id=state["run_id"],
                agent_id=self.agent_id,
                task=task,
                response=response_text,
                embedding=embedding_list,
                tokens_used=tokens_used,
                latency_s=latency_s,
                uncertainty=uncertainty,
                upstream=upstream,
                round_num=state.get("round_num", 0),
            )

        return state
