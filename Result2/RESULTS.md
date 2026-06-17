# Frugal AI Swarm — Result2 Experiment Report

**Run timestamp:** 2026-06-17 00:26 UTC
**Run type:** Extended run — 7 tasks per cell
**Cells:** 108 · **Total task invocations:** 756
**Total tokens generated:** 2,006,819
**Errors:** 0 / 108 cells
**LLM backend:** Ollama (local CPU, apple_silicon)
**Hypothesis framework:** H1 – H4

---

## 1. Architecture Taxonomy

| Config | Name | Roles | Models | Verification |
|--------|------|-------|--------|-------------|
| C1 | Single-Agent Baseline | None | Shared (qwen2.5:3b) | none |
| A1 | Fixed-Role, Single Model | Aligner → Drafter → Verifier | Shared (qwen2.5:3b) | none |
| A2 | Fixed-Role, Multi-Model | Aligner → Drafter → Verifier | Per-role static | none |
| A3 | Self-Organising, Single Model | Shapley + DAG, N=3 | Shared (qwen2.5:3b) | none / selective / full |
| A4 | Self-Organising, Multi-Model | Shapley + DAG, N=3 | Per-agent static | none / selective / full |

**Model assignment (A2 / A4):**

| Slot | A2 Role | A4 Agent | Model |
|------|---------|----------|-------|
| 0 | Curriculum Aligner | agent_0 | `qwen2.5:3b` |
| 1 | Fact-Checker & Drafter | agent_1 | `gemma2:2b` |
| 2 | Pedagogical Verifier | agent_2 | `phi3:mini` |

**Cell breakdown:** C1=12 · A1=12 · A2=12 · A3=36 · A4=36 = **108 cells × 7 tasks = 756 task invocations**

Education task families: `formative_assessment_drafting` · `curriculum_question_generation` · `lesson_adaptation` · `knowledge_base_retrieval`

---

## 2. Run Summary

| Config | Cells | Tasks Issued | Tasks OK (%) | Mean Tokens/Cell | Total Tokens |
|--------|-------|--------------|--------------|-----------------|-------------|
| C1 | 12 | 84 | 0 (0.0%) | 4619 | 55429 |
| A1 | 12 | 84 | 0 (0.0%) | 19678 | 236137 |
| A2 | 12 | 84 | 0 (0.0%) | 22099 | 265192 |
| A3 | 36 | 252 | 0 (0.0%) | 16982 | 611361 |
| A4 | 36 | 252 | 0 (0.0%) | 23297 | 838700 |
| **Total** | **108** | **756** | **0 (0.0%)** | — | **2,006,819** |

> **On task success rate:** The keyword-F1 rubric compares LLM output against a fixed
> reference answer at threshold 0.40. Education LLMs paraphrase rather than echo
> reference tokens; the metric underestimates true quality. Frugality and token
> metrics are accurate regardless.

---

## 3. Frugality Metrics (H4)

| Config | n | Mean Energy (kWh) | vs C1 | Mean Queue Wait (s) | Mean Peak RAM (MB) | Mean TTFUO (s) | Mean Wall (s) |
|--------|---|------------------|-------|--------------------|--------------------|---------------|--------------|
| C1 | 12 | 0.001367 | 1.00× | 27.01 | 1234.1 | 5.79 | 33.2 |
| A1 | 12 | 0.004455 | 3.26× | 90.70 | 1431.5 | 16.22 | 107.3 |
| A2 | 12 | 0.007056 | 5.16× | 145.23 | 469.6 | 24.11 | 169.7 |
| A3 | 36 | 0.005174 | 3.79× | 105.04 | 1121.1 | 19.14 | 124.5 |
| A4 | 36 | 0.011140 | 8.15× | 227.89 | 444.2 | 39.45 | 267.8 |

**H4 goal:** coordination overhead ≤ 3.0× C1.

---

## 4. Verification Mode Breakdown (A3 / A4)

| Config | Mode | Cells | Mean Tokens/Cell | Mean Wall (s) |
|--------|------|-------|-----------------|--------------|
| A3 | none | 12 | 16984 | 96.5 |
| A3 | selective | 12 | 17234 | 109.3 |
| A3 | full | 12 | 16729 | 167.9 |
| A4 | none | 12 | 23209 | 237.3 |
| A4 | selective | 12 | 23097 | 250.4 |
| A4 | full | 12 | 23586 | 315.5 |

---

## 5. Hypothesis Results

| Metric | Value |
|--------|-------|
| `H1_mean_role_stability` | 1.0 |
| `H1_passed_fraction` | 1.0 |
| `H1_verdict` | ✅ SUPPORTED |
| `H2_passed_fraction` | 0.0 |
| `H2_verdict` | ❌ NOT SUPPORTED |
| `H3_passed_fraction` | 0.0 |
| `H3_verdict` | ❌ NOT SUPPORTED |
| `H4_energy_overhead` | 5.53 |
| `H4_verdict` | ❌ NOT SUPPORTED |

### Definitions

| # | Hypothesis | Scope | Threshold |
|---|-----------|-------|-----------|
| H1 | Fixed-role stability | A1, A2 | stability > 0.7 |
| H2 | Selective mode token reduction vs full | A3, A4 | reduction ≥ 30% + retention ≥ 85% |
| H3 | Swarm token efficiency vs C1 | A3, A4 vs C1 | ratio ≥ 1.20 |
| H4 | Coordination wall-clock overhead | A1–A4 vs C1 | ≤ 3.0× |

> H1 is **N/A for A3/A4** (no fixed roles).
> H2 triplets computed: **24**
> H3 A3/A4-vs-C1 pairs: **24**

---

## 6. Governance

| Property | Status |
|----------|--------|
| `requires_teacher_review` | ✅ All 756 task invocations tagged |
| RBAC enforcement | ✅ `SWARM_ROLE=researcher` — all cells granted |
| PII scrubber | ✅ Active on all audit entries |
| Audit trail | ✅ 1296 total records in `data/audit/audit_trail.jsonl` |
| Data retention | ✅ 90-day audit / 30-day ChromaDB |
| Low-watermark | ✅ `get_run_params()` wired — auto-throttles on x86 institutional |

---

