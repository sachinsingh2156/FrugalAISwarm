"""
Experimental configurations — A1-A4 taxonomy (education-focused).

C1   — Single-Agent Baseline: one agent, no swarm.
A1   — Fixed-Role, Single Model: Aligner → Drafter → Verifier; one model for all.
A2   — Fixed-Role, Multi-Model: same 3 roles; one *different* model per role
        (qwen2.5:3b → gemma2:2b → phi3:mini), statically assigned.
A3   — Self-Organising, Single Model: no fixed roles; agents negotiate via
        Shapley + DAG; one model for all 3 agents.
A4   — Self-Organising, Multi-Model: same negotiation mechanism as A3; each of
        the 3 agents runs a different model from MODEL_POOL.

All 4 use swarm_size=3 for a fair comparison.  C1 is the baseline for H4
(token/latency overhead).

Education task families apply to ALL configurations, enabling a clean
within-domain comparison across all 4 architectures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import VerificationMode, SEEDS, MODEL_POOL

# ── Education fixed-role prompts (A1 / A2) ───────────────────────────────────
# Roles mirror a real curriculum-design team so that the swarm's coordination
# gain can be measured against a known-role, education-specific baseline.
CURRICULUM_ALIGNER_PROMPT = (
    "You are the Curriculum Aligner. Your role is to map the task to the "
    "relevant learning objectives, curriculum standards, and learner level. "
    "Identify what prior knowledge is assumed and what new knowledge should "
    "be developed. Do NOT produce final content yet — focus on alignment."
)
FACT_CHECKER_PROMPT = (
    "You are the Fact-Checker and Content Drafter. Working from the alignment "
    "provided, draft the educational content (questions, assessment items, or "
    "lesson notes as required). Ensure all facts are accurate, terminology is "
    "correct, and the level is appropriate. Flag any uncertainty."
)
PEDAGOGICAL_VERIFIER_PROMPT = (
    "You are the Pedagogical Verifier. Review the drafted content for "
    "pedagogical quality: clarity, cognitive load, inclusivity, and alignment "
    "with the stated learning objectives. Output the final, polished version. "
    "Note: this output is for educator review only — no automated high-stakes "
    "assessment decisions will be made from it."
)

EDU_ROLE_PROMPTS = [
    CURRICULUM_ALIGNER_PROMPT,
    FACT_CHECKER_PROMPT,
    PEDAGOGICAL_VERIFIER_PROMPT,
]

EDUCATION_FAMILIES = [
    "formative_assessment_drafting",
    "curriculum_question_generation",
    "lesson_adaptation",
    "knowledge_base_retrieval",
]


@dataclass
class ExperimentConfig:
    name: str               # "C1" | "A1" | "A2" | "A3" | "A4"
    swarm_size: int         # 1 or 3
    use_roles: bool         # True for A1/A2 (fixed pipeline roles)
    role_prompts: list[str] = field(default_factory=list)
    # Static per-role (A2) or per-agent (A4) model list from MODEL_POOL.
    # None → all agents use the shared OllamaClient model (single-model configs).
    model_assignment: list[str] | None = None
    verification_modes: list[VerificationMode] = field(
        default_factory=lambda: list(VerificationMode)
    )
    seeds: list[int] = field(default_factory=lambda: SEEDS)
    families: list[str] = field(default_factory=lambda: EDUCATION_FAMILIES)


# ── Configurations ────────────────────────────────────────────────────────────

# Baseline: single agent, education families, no coordination overhead.
C1 = ExperimentConfig(
    name="C1",
    swarm_size=1,
    use_roles=False,
    verification_modes=[VerificationMode.NONE],
)

# A1: Fixed-Role, Single Model
# Aligner → Drafter → Verifier; one shared model.
A1 = ExperimentConfig(
    name="A1",
    swarm_size=3,
    use_roles=True,
    role_prompts=EDU_ROLE_PROMPTS,
    verification_modes=[VerificationMode.NONE],
)

# A2: Fixed-Role, Multi-Model
# Same 3 roles as A1; each role runs a different model from MODEL_POOL.
# Aligner=qwen2.5:3b, Drafter=gemma2:2b, Verifier=phi3:mini.
A2 = ExperimentConfig(
    name="A2",
    swarm_size=3,
    use_roles=True,
    role_prompts=EDU_ROLE_PROMPTS,
    model_assignment=list(MODEL_POOL),
    verification_modes=[VerificationMode.NONE],
)

# A3: Self-Organising, Single Model
# No fixed roles; Shapley + DAG negotiation; one shared model.
A3 = ExperimentConfig(
    name="A3",
    swarm_size=3,
    use_roles=False,
    verification_modes=list(VerificationMode),
)

# A4: Self-Organising, Multi-Model
# Same negotiation mechanism as A3; agent_0→qwen2.5:3b, agent_1→gemma2:2b,
# agent_2→phi3:mini (static, not bid-based).
A4 = ExperimentConfig(
    name="A4",
    swarm_size=3,
    use_roles=False,
    model_assignment=list(MODEL_POOL),
    verification_modes=list(VerificationMode),
)

ALL_CONFIGS = [C1, A1, A2, A3, A4]


def enumerate_cells(configs: list[ExperimentConfig] | None = None) -> list[dict[str, Any]]:
    """
    Return all experimental cells as a flat list of dicts.
    Each cell = {config, family, seed, verification_mode}.

    C1  × 4 edu families × 3 seeds × 1 mode  = 12 cells
    A1  × 4 edu families × 3 seeds × 1 mode  = 12 cells
    A2  × 4 edu families × 3 seeds × 1 mode  = 12 cells
    A3  × 4 edu families × 3 seeds × 3 modes = 36 cells
    A4  × 4 edu families × 3 seeds × 3 modes = 36 cells
    Total = 108 cells
    """
    if configs is None:
        configs = ALL_CONFIGS
    cells = []
    for cfg in configs:
        for family in cfg.families:
            for seed in cfg.seeds:
                for mode in cfg.verification_modes:
                    cells.append({
                        "config": cfg,
                        "family": family,
                        "seed": seed,
                        "verification_mode": mode,
                    })
    return cells
