"""
Verification mode dispatcher.

Wraps the three verification behaviours as a callable that the agent node
can use as its verification_fn:
  - full:      anti-conformity debate (FREE-MAD) + MAV binary voting
  - selective: only verify if uncertainty signal exceeds threshold
  - none:      pass through (baseline)

Usage:
    verifier = make_verifier(mode=VerificationMode.FULL, client=client)
    verified, result_str = verifier(task, response_text)
"""
from __future__ import annotations

from typing import Callable, Any

from frugal_swarm.coordination.state import Task
from frugal_swarm.model.ollama_client import OllamaClient
from frugal_swarm.verification.mav_verifier import MAVVerifier
from frugal_swarm.verification.debate import AntiConformityDebate
from frugal_swarm.verification.uncertainty import UncertaintyNormaliser

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import VerificationMode


VerificationFn = Callable[[Task, str], tuple[bool, str]]


def make_verifier(
    mode: VerificationMode | str,
    client: OllamaClient,
    normaliser: UncertaintyNormaliser | None = None,
) -> VerificationFn | None:
    """
    Build and return the verification callable for the given mode.
    Returns None for NONE mode (no verification overhead).

    The returned callable signature is:
        fn(task: Task, response: str) -> (verified: bool, result_str: str)

    Note: the uncertainty_raw is not accessible here because it's computed
    inside the agent node.  For SELECTIVE mode the agent node calls
    verifier only when uncertainty exceeds the threshold.
    (See agent_node.py — the verifier callable is invoked only when needed.)
    """
    mode = VerificationMode(mode)

    if mode == VerificationMode.NONE:
        return None

    mav = MAVVerifier(client=client)
    debate = AntiConformityDebate(client=client)

    if mode == VerificationMode.FULL:
        def full_verifier(task: Task, response: str) -> tuple[bool, str]:
            # Step 1: anti-conformity debate → improved answer
            debate_result = debate.debate(task, response)
            improved = debate_result.final_answer

            # Step 2: MAV binary voting on improved answer
            mav_result = mav.verify(task, improved)
            summary = (
                f"[DEBATE] {debate_result.rounds} round(s). "
                f"[MAV] {mav_result.summary}"
            )
            return mav_result.passed, summary

        return full_verifier

    if mode == VerificationMode.SELECTIVE:
        if normaliser is None:
            normaliser = UncertaintyNormaliser()

        def selective_verifier(task: Task, response: str) -> tuple[bool, str]:
            # Selective verification is triggered from agent_node when
            # uncertainty >= threshold.  Here we just run MAV (no debate for
            # speed — debate is saved for full mode).
            mav_result = mav.verify(task, response)
            return mav_result.passed, f"[SELECTIVE-MAV] {mav_result.summary}"

        return selective_verifier

    return None


def selective_gate(
    task: Task,
    response: str,
    uncertainty_raw: float | None,
    normaliser: UncertaintyNormaliser,
    verifier: VerificationFn,
) -> tuple[bool, str | None]:
    """
    Helper called by agent_node in SELECTIVE mode:
    only invoke verifier if uncertainty exceeds threshold.
    """
    if normaliser.should_verify(task.family, uncertainty_raw):
        verified, result = verifier(task, response)
        return verified, result
    return False, None  # not verified, no overhead
