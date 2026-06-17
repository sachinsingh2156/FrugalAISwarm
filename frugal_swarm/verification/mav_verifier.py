"""
Multi-Agent Verification (MAV) — Lifshitz, McIlraith & Du (2025).

Each Aspect Verifier is prompted to produce a binary True/False signal for a
specific property of the candidate output.  Signals are aggregated by
majority vote.

Verifiers are drawn from the SAME identical-agent pool (no role labels) to
preserve the no-role-label property of the swarm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frugal_swarm.model.ollama_client import OllamaClient
from frugal_swarm.coordination.state import Task

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import N_VERIFIERS


# ── Aspect definitions ────────────────────────────────────────────────────────

ASPECTS = [
    {
        "name": "factual_accuracy",
        "prompt_template": (
            "Is the following answer factually accurate given the question?\n"
            "Question: {question}\n"
            "Answer: {answer}\n\n"
            "Reply with ONLY 'True' or 'False'."
        ),
    },
    {
        "name": "relevance",
        "prompt_template": (
            "Does the following answer directly address the question asked?\n"
            "Question: {question}\n"
            "Answer: {answer}\n\n"
            "Reply with ONLY 'True' or 'False'."
        ),
    },
    {
        "name": "completeness",
        "prompt_template": (
            "Is the following answer reasonably complete (covers the key points)?\n"
            "Question: {question}\n"
            "Answer: {answer}\n\n"
            "Reply with ONLY 'True' or 'False'."
        ),
    },
]


@dataclass
class VerifierVote:
    aspect: str
    verifier_id: int
    verdict: bool
    raw_response: str
    tokens_used: int


@dataclass
class MAVResult:
    passed: bool                    # majority vote across all aspect-verifier pairs
    votes: list[VerifierVote]
    aspect_scores: dict[str, float] # fraction of True votes per aspect
    total_tokens: int
    summary: str


def _parse_bool(text: str) -> bool:
    return text.strip().lower().startswith("true")


class MAVVerifier:
    """
    Runs N_VERIFIERS verifier calls per aspect and aggregates by majority vote.
    """

    def __init__(
        self,
        client: OllamaClient,
        n_verifiers: int = N_VERIFIERS,
        aspects: list[dict] | None = None,
        temperature: float = 0.3,
    ) -> None:
        self.client = client
        self.n_verifiers = n_verifiers
        self.aspects = aspects or ASPECTS
        self.temperature = temperature

    def verify(self, task: Task, candidate: str) -> MAVResult:
        """
        Run all aspect verifiers against (task.prompt, candidate).
        Returns a MAVResult with the aggregate verdict.
        """
        votes: list[VerifierVote] = []
        total_tokens = 0

        for aspect in self.aspects:
            prompt_text = aspect["prompt_template"].format(
                question=task.prompt,
                answer=candidate,
            )
            for v_idx in range(self.n_verifiers):
                try:
                    resp = self.client.generate(
                        prompt=prompt_text,
                        temperature=self.temperature,
                        max_tokens=8,
                    )
                    verdict = _parse_bool(resp["text"])
                    total_tokens += resp["tokens_used"]
                except Exception:
                    verdict = False
                    total_tokens += 0

                votes.append(
                    VerifierVote(
                        aspect=aspect["name"],
                        verifier_id=v_idx,
                        verdict=verdict,
                        raw_response=resp["text"] if "resp" in dir() else "ERROR",
                        tokens_used=resp["tokens_used"] if "resp" in dir() else 0,
                    )
                )

        # Per-aspect scores
        aspect_scores: dict[str, float] = {}
        for aspect in self.aspects:
            a_votes = [v.verdict for v in votes if v.aspect == aspect["name"]]
            aspect_scores[aspect["name"]] = sum(a_votes) / len(a_votes) if a_votes else 0.0

        # Overall pass = majority across ALL votes
        all_verdicts = [v.verdict for v in votes]
        passed = sum(all_verdicts) / len(all_verdicts) > 0.5 if all_verdicts else False

        passing_aspects = [k for k, s in aspect_scores.items() if s > 0.5]
        summary = (
            f"{'PASS' if passed else 'FAIL'}: "
            f"{sum(all_verdicts)}/{len(all_verdicts)} votes positive. "
            f"Passing aspects: {passing_aspects or 'none'}."
        )

        return MAVResult(
            passed=passed,
            votes=votes,
            aspect_scores=aspect_scores,
            total_tokens=total_tokens,
            summary=summary,
        )
