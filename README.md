# Frugal AI Agent Swarms — Phase-1 Pilot

**Self-Organising and Verification-Aware Multi-Agent Systems for Edge Infrastructure**

This repository contains the full Phase-1 pilot implementation as described in the research synopsis, incorporating all supervisor feedback from Ricky Cheng (COL).
Platform: x86 institutional server / Apple Silicon · Stack: Python 3.11 · LangGraph · Ollama · ChromaDB · MLflow

---

## Quick start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama and pull models (1B–4B, CPU-only, no GPU required)
ollama serve &
ollama pull qwen2.5:1.5b   # Agent-0: reasoning / maths (1.5B)
ollama pull llama3.2:3b    # Agent-1: extraction / summarisation (3B)
ollama pull phi3:mini      # Agent-2: structured planning (3.8B, Microsoft)

# 3. Day-2 model benchmark (auto-selects the best model)
python run_experiment.py benchmark

# 4. Quick smoke test (1 task per cell, ~5 min)
python run_experiment.py run --smoke

# 5. Single demo run (10 tasks, C3 N=3, no verification)
python run_experiment.py demo --max-tasks 10

# 6. Full experiment sweep (all cells, ~hours)
python run_experiment.py run

# 7. Open the demo notebook
jupyter lab demo_notebook.ipynb
```

## Quick start (Docker — dashboard only)

Ollama must be running locally first (`ollama serve` or the macOS app).

```bash
cp .env.example .env
# For Docker dashboard, set in .env: OLLAMA_BASE_URL=http://host.docker.internal:11434
make pull-models              # pull LLMs into local Ollama
make up-d                     # start dashboard container
open http://localhost:5050
```

See [DEPLOY.md](DEPLOY.md) for full deployment guide including low-watermark mode.

### Visual dashboard

```bash
python dashboard_server.py
# → open http://localhost:5050
```

The dashboard streams the full swarm workflow in real time — task distribution, per-agent responses, Shapley scoring, DAG routing, MAV verification, C4 capability bidding, and the final combined output — for all four configurations (C2, C2-EDU, C3, C4).

---

## Project structure

```
autonomousSwam/
├── config.py                     # Central configuration (model, thresholds, paths)
├── run_experiment.py             # CLI entry point
├── dashboard_server.py           # Flask SSE server for the visual dashboard
├── dashboard.html                # Single-file dashboard frontend (markdown rendering)
├── requirements.txt
├── demo_notebook.ipynb           # End-to-end demo (C3 N=3, multistep_qa)
├── Dockerfile                    # Container image for dashboard
├── docker-compose.yml            # One-command stack (Ollama + Dashboard)
├── Makefile                      # make up / down / pull-models / test
├── DEPLOY.md                     # Deployment guide
├── .env.example                  # Environment variable template
│
├── frugal_swarm/
│   ├── model/
│   │   ├── ollama_client.py      # Ollama HTTP client + model_override support
│   │   ├── model_benchmark.py    # Day-2 auto-select (reliability floor)
│   │   └── embedder.py           # all-MiniLM-L6-v2 + Shapley score computation
│   │
│   ├── coordination/
│   │   ├── state.py              # SwarmState (LangGraph typed dict) + Task dataclass
│   │   ├── task_board.py         # Shared task board (atomic claim / submit)
│   │   ├── agent_node.py         # Identical agent node (NO role labels — RQ1)
│   │   ├── shapley.py            # Rolling Shapley estimator (SELFORG method)
│   │   ├── dag_builder.py        # Response-conditioned DAG builder
│   │   ├── role_monitor.py       # Role-stability + specialisation-entropy (H1)
│   │   └── capability_router.py  # C4: model assignment, decomposition, bidding
│   │
│   ├── graph/
│   │   └── swarm_graph.py        # LangGraph StateGraph with SQLite checkpointer
│   │
│   ├── verification/
│   │   ├── uncertainty.py        # Token log-prob uncertainty signal
│   │   ├── mav_verifier.py       # MAV aspect verifiers (binary voting)
│   │   ├── debate.py             # FREE-MAD anti-conformity debate
│   │   └── modes.py              # full / selective / none dispatcher
│   │
│   ├── memory/
│   │   ├── chroma_store.py       # ChromaDB: traces, scores, tasks
│   │   └── mlflow_tracker.py     # MLflow experiment + hypothesis metrics
│   │
│   ├── corpus/
│   │   ├── loader.py             # Task corpus loader (all 7 families)
│   │   ├── rubrics.py            # Scoring rubrics per family (incl. education)
│   │   └── tasks/
│   │       ├── multistep_qa.json               # 50 tasks (general)
│   │       ├── document_analysis.json           # 50 tasks (general)
│   │       ├── workflow_planning.json           # 50 tasks (general)
│   │       ├── formative_assessment_drafting.json  # 50 tasks (edu, high_stakes=false)
│   │       ├── curriculum_question_generation.json # 50 tasks (edu, high_stakes=false)
│   │       ├── lesson_adaptation.json              # 50 tasks (edu, high_stakes=false)
│   │       └── knowledge_base_retrieval.json       # 50 tasks (edu, high_stakes=false)
│   │
│   ├── experiments/
│   │   ├── configurations.py     # C1/C2/C2_EDU/C3_N3/C3_N5/C4_N3 configs
│   │   ├── runner.py             # Multi-cell experiment runner
│   │   └── metrics.py            # H1–H4 metric computation
│   │
│   ├── metrics/
│   │   ├── frugality.py          # H4: wall_clock, TTFUO, RAM, energy, queue_wait
│   │   └── hardware.py           # Platform detection, low-watermark mode
│   │
│   ├── governance/
│   │   ├── pii_scrubber.py       # Regex-based PII redaction (email, phone, NI, etc.)
│   │   ├── audit_trail.py        # Append-only JSONL audit log (all agent actions)
│   │   ├── retention.py          # Data retention policy enforcement (90-day default)
│   │   └── rbac.py               # Role-based access control stubs
│   │
│   ├── analysis/
│   │   └── qualitative.py        # Codebook + Cohen's κ + Fleiss' κ + CSV loader
│   │
│   └── dashboard/
│       └── runner.py             # Step-by-step event generators (run_c2/c2_edu/c3/c4)
│
└── analysis/
    ├── hypothesis_table.py       # Print H1–H4 results from MLflow
    └── qualitative_coding.py     # RQ1 trace coding tool (≥ 500 traces)
```

---

## Experimental design

| Config | Description | Swarm size | Families |
|--------|-------------|------------|---------|
| C1 | Single-agent baseline | 1 | General |
| C2 | Fixed-role (Planner → Executor → Verifier) | 3 | General |
| **C2-EDU** | **Education benchmark** (Aligner → Drafter → Verifier) | 3 | **Education** |
| C3 N=3 | Self-organising, no role labels | 3 | General |
| C3 N=5 | Self-organising, no role labels | 5 | General |
| C4 N=3 | Heterogeneous team — each agent owns a different LLM | 3 | General |

**C2-EDU** provides the education-domain fixed-role baseline against which C3/C4 coordination gains are measured (supervisor recommendation: benchmark against known-role, in-domain agents).

### Education task corpus (4 new families)

All education families have `high_stakes: false` — no automated high-stakes assessment decisions are made from swarm outputs; all results are intended for educator review only.

| Family | Description |
|--------|-------------|
| `formative_assessment_drafting` | Draft formative assessment items aligned to learning objectives |
| `curriculum_question_generation` | Generate curriculum-aligned questions at specified cognitive levels |
| `lesson_adaptation` | Adapt lesson plans for different learner groups or modalities |
| `knowledge_base_retrieval` | Retrieve factual content from a simulated institution KB |

### C4 — Heterogeneous-model team

Each agent is assigned a **different LLM** from `MODEL_POOL` in `config.py` (default: `qwen2.5:1.5b`, `llama3.2:3b`, `phi3:mini`). When a task arrives:

1. A decomposer LLM splits the task into typed sub-tasks (`reasoning`, `calculation`, `extraction`, `summarisation`, `planning`, `synthesis`).
2. Agents **competitively bid** — bid score = `CAPABILITY_PROFILES[model][sub-type]` + small noise.
3. The highest bidder for each sub-task wins and executes it with their own model.
4. The agent with the highest `synthesis` score acts as **synthesiser**, merging all partial answers into one final response.

Edit `MODEL_POOL` and `CAPABILITY_PROFILES` in `config.py` to match the models you have in Ollama.

### Verification modes

| Mode | Behaviour |
|------|-----------|
| `none` | Output emitted as-is (baseline) |
| `selective` | Only high-uncertainty outputs verified (MAV) |
| `full` | ALL outputs through FREE-MAD debate + MAV voting |

### Hypotheses

| # | Threshold | Metric |
|---|-----------|--------|
| H1 | Role stability > 70 % | `mean_role_stability` in MLflow |
| H2 | Token reduction ≥ 30 %, reliability retention ≥ 85 % | `h2_token_reduction`, `h2_reliability_retention` |
| H3 | Token efficiency ratio ≥ 1.2× | `h3_efficiency_ratio` |
| **H4** | **Swarm wall-clock ≤ 3× C1 baseline** | `frugality/wall_clock_s`, `frugality/ttfuo_s`, `frugality/peak_ram_mb`, `frugality/energy_kwh_est` |

---

## Frugality metrics (H4)

The `frugal_swarm/metrics/frugality.py` module collects per-run resource metrics and logs them to MLflow:

| Metric | Description |
|--------|-------------|
| `wall_clock_s` | Total wall-clock time (task_started → run_complete) |
| `ttfuo_s` | Time-To-First-Useful-Output (seconds to first agent response) |
| `peak_ram_mb` | Peak RSS memory during the run (psutil) |
| `energy_kwh_est` | Energy estimate via TDP proxy (CPU-only, no GPU assumed) |
| `queue_wait_s` | Cumulative agent wait time (coordination overhead) |

**Low-watermark mode** (set `LOW_WATERMARK_MODE=1`) activates reduced parameters suitable for the x86 institutional server baseline (dual Xeon, 64 GB RAM, no GPU):

```bash
LOW_WATERMARK_MODE=1 python dashboard_server.py
# or: python frugal_swarm/metrics/hardware.py   # print hardware profile
```

---

## Governance

All agent inputs and outputs are processed through `frugal_swarm/governance/`:

- **PII scrubbing** — email, phone, NI number, student IDs, postcodes, DOBs redacted before any logging
- **Audit trail** — append-only JSONL at `data/audit/audit_trail.jsonl` (every agent action)
- **Retention policy** — records older than 90 days archived; configurable via `AUDIT_RETENTION_DAYS`
- **RBAC** — role-based access (`researcher` / `educator` / `admin` / `readonly`); set `SWARM_ROLE` env var

---

## Qualitative coding + inter-rater reliability

```bash
# Run the codebook CLI
python -m frugal_swarm.analysis.qualitative

# Compare two raters' CSV and compute Cohen's κ
python -m frugal_swarm.analysis.qualitative ratings.csv
python -m frugal_swarm.analysis.qualitative ratings.csv --json

# Three raters → also computes Fleiss' κ
```

Expected CSV format:

```
item_id,rater_1,rater_2
item_001,CODE_ROLE_EMERGENCE,CODE_ROLE_EMERGENCE
item_002,CODE_CONSENSUS_REACHED,CODE_CONFLICT_RESOLVED
```

Codebook codes: `CODE_ROLE_EMERGENCE`, `CODE_CONSENSUS_REACHED`, `CODE_CONFLICT_RESOLVED`, `CODE_REDUNDANT_RESPONSE`, `CODE_VERIFICATION_USEFUL`, `CODE_FRUGAL_SHORTCUT`

Interpretation follows Landis & Koch (1977): κ ≥ 0.61 = Substantial, κ ≥ 0.81 = Almost perfect.

---

## Viewing results

```bash
# MLflow UI
mlflow ui --backend-store-uri data/mlflow

# Hypothesis table (H1–H4)
python analysis/hypothesis_table.py

# Qualitative trace coding (auto-code)
python analysis/qualitative_coding.py

# Interactive coding
python analysis/qualitative_coding.py --interactive

# Hardware profile
python frugal_swarm/metrics/hardware.py

# Audit trail retention (dry run)
python frugal_swarm/governance/retention.py --dry-run
```

---

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_CANDIDATES` | `[qwen2.5:1.5b, llama3.2:3b]` | Small models benchmarked on Day 2 (1B–3B) |
| `MODEL_POOL` | 3 models | C4 heterogeneous team pool |
| `DEFAULT_VERIFICATION_MODE` | `selective` | Verification mode |
| `UNCERTAINTY_THRESHOLD` | `0.5` | Selective verification trigger |
| `H4_WALL_OVERHEAD_THRESHOLD` | `3.0` | Max swarm/C1 wall-clock ratio |
| `LOW_WATERMARK_MODE` | `False` | Activate reduced-resource mode |
| `AUDIT_RETENTION_DAYS` | `90` | Audit log retention window |
| `CHROMA_RETENTION_DAYS` | `30` | ChromaDB retention window |

---

## Phase-1 milestones

| Milestone | Day | Criterion |
|-----------|-----|-----------|
| M1 | D3 | Ollama + embedding pipeline pass 10-pair smoke test |
| M2 | D9 | 10-task N=3 dry-run: claims, Shapley, DAG, traces all visible |
| M3 | D14 | C1/C2/C3 full runs complete with matched token budgets |
| M4 | D16 | Verification sweep done; H2 quantities computable |
| M5 | D20 | Report + repo + ≥500 coded traces + demo notebook delivered |

---

## References

- SELFORG (Tastan et al., 2026) — Shapley-based contribution estimation
- FREE-MAD (Cui et al., 2025) — anti-conformity debate
- MAV (Lifshitz et al., 2025) — multi-aspect binary verification
- EdgeShard (Zhang et al., 2024) — edge LLM sharding (Phase 3)
- LangGraph (LangChain, 2026)
- Landis & Koch (1977) — kappa interpretation benchmarks
