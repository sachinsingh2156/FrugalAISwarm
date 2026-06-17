"""
FREE-MAD Anti-Conformity Debate — Cui et al. (2025).

Verifiers are explicitly prompted to look for what the candidate got WRONG
rather than to seek consensus.  This prevents the conformity failure mode
where agents converge on a plausible-sounding but incorrect answer.

One debate round is sufficient for a 16–19 % accuracy improvement
(per FREE-MAD paper) while keeping token costs low.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frugal_swarm.model.ollama_client import OllamaClient
from frugal_swarm.coordination.state import Task

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DEBATE_ROUNDS, N_VERIFIERS


ANTI_CONFORMITY_PROMPT = """\
You are a critical reviewer. Your job is to identify what is WRONG or missing
in the following answer — do NOT simply agree with it.

Question: {question}

Candidate answer: {answer}

Identify any factual errors, missing key points, or logical flaws.
Then provide an improved answer (or confirm it is correct if you find no issues).

Format:
CRITIQUE: <your critique>
IMPROVED ANSWER: <your improved answer or 'No change needed'>"""


SCORE_AGGREGATION_PROMPT = """\
Below are {n} alternative answers to the following question:

Question: {question}

{answers}

Select the BEST answer based on accuracy and completeness.
Output ONLY the number of the best answer (1, 2, 3, …)."""


@dataclass
class DebateResult:
    final_answer: str
    critiques: list[str]
    improved_answers: list[str]
    total_tokens: int
    rounds: int


class AntiConformityDebate:
    """
    Runs one FREE-MAD anti-conformity debate round.
    """

    def __init__(
        self,
        client: OllamaClient,
        n_debaters: int = N_VERIFIERS,
        rounds: int = DEBATE_ROUNDS,
        temperature: float = 0.5,
    ) -> None:
        self.client = client
        self.n_debaters = n_debaters
        self.rounds = rounds
        self.temperature = temperature

    def debate(self, task: Task, candidate: str) -> DebateResult:
        """
        Run anti-conformity debate on (task, candidate).
        Returns the aggregated best answer.
        """
        total_tokens = 0
        current_answer = candidate
        all_critiques: list[str] = []
        all_improved: list[str] = []

        for _round in range(self.rounds):
            improved_answers: list[str] = []

            for _ in range(self.n_debaters):
                prompt = ANTI_CONFORMITY_PROMPT.format(
                    question=task.prompt,
                    answer=current_answer,
                )
                try:
                    resp = self.client.generate(
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=300,
                    )
                    text = resp["text"]
                    total_tokens += resp["tokens_used"]

                    # Parse critique and improved answer
                    critique_part = ""
                    improved_part = current_answer
                    if "CRITIQUE:" in text:
                        parts = text.split("IMPROVED ANSWER:")
                        critique_part = parts[0].replace("CRITIQUE:", "").strip()
                        if len(parts) > 1:
                            improved_part = parts[1].strip()
                            if "No change needed" in improved_part:
                                improved_part = current_answer

                    all_critiques.append(critique_part)
                    improved_answers.append(improved_part)
                    all_improved.append(improved_part)

                except Exception as exc:
                    improved_answers.append(current_answer)

            # Score aggregation: pick the best improved answer
            if len(improved_answers) == 1:
                current_answer = improved_answers[0]
            elif improved_answers:
                candidates_text = "\n".join(
                    f"{i+1}. {ans}" for i, ans in enumerate(improved_answers)
                )
                agg_prompt = SCORE_AGGREGATION_PROMPT.format(
                    n=len(improved_answers),
                    question=task.prompt,
                    answers=candidates_text,
                )
                try:
                    agg_resp = self.client.generate(
                        agg_prompt, temperature=0.0, max_tokens=8
                    )
                    total_tokens += agg_resp["tokens_used"]
                    idx_str = agg_resp["text"].strip().split()[0]
                    idx = int(idx_str) - 1
                    if 0 <= idx < len(improved_answers):
                        current_answer = improved_answers[idx]
                    else:
                        current_answer = improved_answers[0]
                except Exception:
                    current_answer = improved_answers[0]

        return DebateResult(
            final_answer=current_answer,
            critiques=all_critiques,
            improved_answers=all_improved,
            total_tokens=total_tokens,
            rounds=self.rounds,
        )
