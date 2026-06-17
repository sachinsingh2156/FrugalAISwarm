"""
Central configuration for the Frugal AI Agent Swarms Phase-1 Pilot.
Override any value via environment variables or by editing this file.

MODEL CHANGE (supervisor feedback — frugality focus):
  All models are now in the 1B-4B parameter range to match the
  edge / institutional server hardware baseline (no GPU, CPU-only inference).
  qwen2.5:7b  → qwen2.5:1.5b   (reasoning, maths)
  llama3.1:8b → llama3.2:3b    (extraction, summarisation)
  mistral:7b  → phi3:mini      (planning, structured tasks, ~3.8B)
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── Verification mode ─────────────────────────────────────────────────────────
class VerificationMode(str, Enum):
    FULL = "full"
    SELECTIVE = "selective"
    NONE = "none"


# ── Model candidates (auto-benchmarked on Day 2) ──────────────────────────────
# Small models only — frugal edge inference, no GPU required
MODEL_CANDIDATES = [
    "qwen2.5:3b",   # primary: strong reasoning
    "phi3:mini",    # fallback: good structured tasks
    "gemma2:2b",    # second fallback: general comprehension
]
SELECTED_MODEL_FILE = DATA_DIR / ".selected_model"

def get_model() -> str:
    if SELECTED_MODEL_FILE.exists():
        return SELECTED_MODEL_FILE.read_text().strip()
    return MODEL_CANDIDATES[0]


# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── Swarm parameters ──────────────────────────────────────────────────────────
DEFAULT_SWARM_SIZE: int = 3
MAX_ROUNDS: int = 2          # C3: 2 rounds (independent → context-aware)
ROLE_STABILITY_WINDOW: int = 10   # lowered from 100 for pilot scale (50 tasks/cell)

# ── Verification ──────────────────────────────────────────────────────────────
DEFAULT_VERIFICATION_MODE: VerificationMode = VerificationMode.SELECTIVE
UNCERTAINTY_THRESHOLD: float = 0.5
N_VERIFIERS: int = 3
DEBATE_ROUNDS: int = 1

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = str(DATA_DIR / "chromadb")
CHROMA_TRACES_COLLECTION: str = "agent_traces"
CHROMA_SCORES_COLLECTION: str = "shapley_scores"
CHROMA_TASKS_COLLECTION: str = "task_definitions"

# ── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", str(DATA_DIR / "mlflow")
)
MLFLOW_EXPERIMENT_NAME: str = "frugal_swarm_phase1"

# ── LangGraph checkpointing ───────────────────────────────────────────────────
LANGGRAPH_DB_PATH: str = str(DATA_DIR / "langgraph_checkpoints.sqlite")

# ── Corpus ────────────────────────────────────────────────────────────────────
CORPUS_DIR: Path = ROOT / "frugal_swarm" / "corpus" / "tasks"
TASKS_PER_FAMILY: int = 50
SEEDS: list[int] = [42, 7, 137]

# ── Scoring thresholds (H1-H3) ────────────────────────────────────────────────
H1_ROLE_STABILITY_THRESHOLD: float = 0.70
H2_TOKEN_REDUCTION_THRESHOLD: float = 0.30
H2_RELIABILITY_RETENTION: float = 0.85
H3_TOKEN_EFFICIENCY_RATIO: float = 1.20

# ── Multi-model pool (A2/A4) ──────────────────────────────────────────────────
# A2: each of the 3 pipeline roles uses a different model (static assignment).
# A4: each of the 3 swarm agents uses a different model (static assignment).
# Edit to match the small models available in your Ollama instance.

MODEL_POOL: list[str] = [
    "qwen2.5:3b",   # slot-0: Curriculum Aligner / agent_0
    "gemma2:2b",    # slot-1: Fact-Checker & Drafter / agent_1
    "phi3:mini",    # slot-2: Pedagogical Verifier / agent_2
]

# ── H4: Frugality (coordination cost) ────────────────────────────────────────
# H4: swarm wall-clock time ≤ H4_WALL_OVERHEAD_THRESHOLD × C1 baseline
# With small models, wall-clock is much lower → threshold is tighter
H4_WALL_OVERHEAD_THRESHOLD: float = 3.0

# Retention windows (days)
AUDIT_RETENTION_DAYS: int = 90
CHROMA_RETENTION_DAYS: int = 30

# ── Hardware / low-watermark mode ─────────────────────────────────────────────
# Set True (or env LOW_WATERMARK_MODE=1) to run on minimal x86 institutional HW
LOW_WATERMARK_MODE: bool = bool(int(os.getenv("LOW_WATERMARK_MODE", "0")))
