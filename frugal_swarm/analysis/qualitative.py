"""
Qualitative coding support with inter-rater reliability.

Implements:
  - Codebook: structured code definitions for swarm behaviour analysis
  - cohen_kappa(): pairwise Cohen's kappa for two raters
  - fleiss_kappa(): multi-rater Fleiss' kappa (N raters, K categories)
  - compare_raters(): CLI-friendly comparison report
  - load_ratings(): parse a CSV of rater judgements

Codebook codes (swarm behaviour)
---------------------------------
CODE_ROLE_EMERGENCE      : agent spontaneously adopts a specialised role
CODE_CONSENSUS_REACHED   : agents converge to a shared answer without conflict
CODE_CONFLICT_RESOLVED   : agents disagree, then resolve through debate/revision
CODE_REDUNDANT_RESPONSE  : agent repeats content already provided by another
CODE_VERIFICATION_USEFUL : verification step materially improves the final answer
CODE_FRUGAL_SHORTCUT     : agent produces correct answer with fewer tokens than peers

Usage
-----
    from frugal_swarm.analysis.qualitative import cohen_kappa, load_ratings

    # From CSV: columns = [item_id, rater_a, rater_b]
    items, a, b = load_ratings("ratings.csv")
    kappa, ci = cohen_kappa(a, b)
    print(f"Cohen's κ = {kappa:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")

CSV format expected by load_ratings():
    item_id,rater_a,rater_b[,rater_c,...]
    item_001,CODE_ROLE_EMERGENCE,CODE_ROLE_EMERGENCE
    item_002,CODE_CONSENSUS_REACHED,CODE_CONFLICT_RESOLVED
    ...
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence


# ── Codebook ──────────────────────────────────────────────────────────────────

CODEBOOK: dict[str, dict] = {
    "CODE_ROLE_EMERGENCE": {
        "label":       "Role Emergence",
        "description": "An agent spontaneously adopts a specialised functional role "
                       "(e.g., summariser, critic, planner) without an explicit role prompt.",
        "indicators":  ["agent uses role-specific language", "consistent task framing across turns"],
        "non_examples": ["agent simply repeats the task prompt"],
    },
    "CODE_CONSENSUS_REACHED": {
        "label":       "Consensus Reached",
        "description": "Two or more agents converge on a shared answer or framing "
                       "without explicit conflict or negotiation.",
        "indicators":  ["similar wording across agents", "no contradictory claims"],
        "non_examples": ["agents giving identical verbatim responses (likely copy)"],
    },
    "CODE_CONFLICT_RESOLVED": {
        "label":       "Conflict Resolved",
        "description": "Agents initially disagree on facts or approach, then one or more "
                       "agents revise their position based on peer input.",
        "indicators":  ["explicit acknowledgement of disagreement", "revision referencing peer"],
        "non_examples": ["agents simply ignoring each other's contradictions"],
    },
    "CODE_REDUNDANT_RESPONSE": {
        "label":       "Redundant Response",
        "description": "An agent's response adds no new information beyond what was "
                       "already provided by an upstream agent.",
        "indicators":  ["high token overlap with prior response", "no novel claims"],
        "non_examples": ["agent deliberately summarises for clarity"],
    },
    "CODE_VERIFICATION_USEFUL": {
        "label":       "Verification Useful",
        "description": "The verification step (MAV or debate) materially improves the "
                       "final answer — corrects an error, adds missing information, or "
                       "increases factual precision.",
        "indicators":  ["pre/post verification quality improvement", "verifier flags genuine error"],
        "non_examples": ["verifier simply restates the original answer"],
    },
    "CODE_FRUGAL_SHORTCUT": {
        "label":       "Frugal Shortcut",
        "description": "An agent produces a correct or high-quality answer using "
                       "significantly fewer tokens than the swarm average.",
        "indicators":  ["token count < 50% of swarm mean", "quality score comparable"],
        "non_examples": ["agent truncates due to error or token limit"],
    },
}

ALL_CODES = list(CODEBOOK.keys())


# ── Cohen's kappa ──────────────────────────────────────────────────────────────

def cohen_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    bootstrap_n: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """
    Compute Cohen's kappa for two raters.

    Parameters
    ----------
    rater_a, rater_b : sequences of string labels (same length)
    bootstrap_n      : number of bootstrap resamples for CI
    ci_level         : confidence level (default 0.95)
    seed             : random seed for reproducibility

    Returns
    -------
    (kappa, (ci_lower, ci_upper))

    Interpretation (Landis & Koch 1977):
        < 0.00  Poor
        0.00–0.20  Slight
        0.21–0.40  Fair
        0.41–0.60  Moderate
        0.61–0.80  Substantial
        0.81–1.00  Almost perfect
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length")
    n = len(rater_a)
    if n == 0:
        raise ValueError("Empty rating lists")

    kappa = _kappa_from_lists(rater_a, rater_b)
    ci = _bootstrap_ci(rater_a, rater_b, bootstrap_n, ci_level, seed)
    return kappa, ci


def _kappa_from_lists(a: Sequence[str], b: Sequence[str]) -> float:
    n = len(a)
    categories = sorted(set(a) | set(b))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for ai, bi in zip(a, b):
        matrix[cat_idx[ai]][cat_idx[bi]] += 1

    # Observed agreement
    po = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    row_sums = [sum(matrix[i]) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row_sums[i] / n) * (col_sums[i] / n) for i in range(k))

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def _bootstrap_ci(
    a: Sequence[str],
    b: Sequence[str],
    n_resamples: int,
    level: float,
    seed: int,
) -> tuple[float, float]:
    import random
    rng = random.Random(seed)
    n = len(a)
    kappas = []
    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        ra = [a[i] for i in indices]
        rb = [b[i] for i in indices]
        kappas.append(_kappa_from_lists(ra, rb))
    kappas.sort()
    alpha = 1.0 - level
    lo = kappas[int(math.floor(alpha / 2 * n_resamples))]
    hi = kappas[min(int(math.ceil((1 - alpha / 2) * n_resamples)), n_resamples - 1)]
    return lo, hi


# ── Fleiss' kappa (multi-rater) ───────────────────────────────────────────────

def fleiss_kappa(ratings_matrix: list[list[int]]) -> float:
    """
    Fleiss' kappa for N raters and K categories.

    ratings_matrix: list of N_items rows; each row is a list of K counts
                    (how many raters assigned each category to that item).

    Returns the kappa value.
    """
    n_items = len(ratings_matrix)
    n_cats = len(ratings_matrix[0])
    n_raters = sum(ratings_matrix[0])

    # Overall proportion for each category
    p_j = [
        sum(ratings_matrix[i][j] for i in range(n_items)) / (n_items * n_raters)
        for j in range(n_cats)
    ]

    # Per-item agreement
    P_i = [
        (sum(ratings_matrix[i][j] ** 2 for j in range(n_cats)) - n_raters)
        / (n_raters * (n_raters - 1))
        for i in range(n_items)
    ]

    P_bar = sum(P_i) / n_items
    P_e_bar = sum(p ** 2 for p in p_j)

    if P_e_bar == 1.0:
        return 1.0
    return (P_bar - P_e_bar) / (1.0 - P_e_bar)


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_ratings(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """
    Load a ratings CSV.

    Expected columns: item_id, rater_1, rater_2[, rater_3, ...]

    Returns
    -------
    (item_ids, rater_columns)
    where rater_columns[i] is the list of ratings for rater i+1.
    """
    path = Path(path)
    item_ids = []
    rater_columns: list[list[str]] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rater_headers = [h for h in headers if h != "item_id"]

        for _ in rater_headers:
            rater_columns.append([])

        for row in reader:
            item_ids.append(row["item_id"])
            for idx, rh in enumerate(rater_headers):
                rater_columns[idx].append(row[rh].strip())

    return item_ids, rater_columns


# ── CLI comparison report ─────────────────────────────────────────────────────

def compare_raters(path: str | Path, output_json: bool = False) -> dict:
    """
    Generate an inter-rater reliability report from a ratings CSV.

    Reports pairwise Cohen's kappa for all rater pairs, plus Fleiss' kappa
    if there are 3+ raters.

    Returns a dict suitable for JSON output or MLflow logging.
    """
    item_ids, rater_cols = load_ratings(path)
    n_raters = len(rater_cols)
    report: dict = {
        "n_items":  len(item_ids),
        "n_raters": n_raters,
        "pairwise": {},
        "fleiss_kappa": None,
    }

    # Pairwise Cohen's kappa
    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            k, ci = cohen_kappa(rater_cols[i], rater_cols[j])
            key = f"rater_{i+1}_vs_rater_{j+1}"
            report["pairwise"][key] = {
                "kappa":    round(k, 4),
                "ci_lower": round(ci[0], 4),
                "ci_upper": round(ci[1], 4),
                "interpretation": _interpret_kappa(k),
            }

    # Fleiss' kappa (3+ raters)
    if n_raters >= 3:
        all_codes = sorted(set(c for col in rater_cols for c in col))
        code_idx = {c: i for i, c in enumerate(all_codes)}
        matrix = []
        for item_i in range(len(item_ids)):
            row = [0] * len(all_codes)
            for col in rater_cols:
                row[code_idx[col[item_i]]] += 1
            matrix.append(row)
        fk = fleiss_kappa(matrix)
        report["fleiss_kappa"] = {
            "kappa": round(fk, 4),
            "interpretation": _interpret_kappa(fk),
        }

    if output_json:
        print(json.dumps(report, indent=2))
    return report


def _interpret_kappa(k: float) -> str:
    if k < 0:
        return "Poor (< 0)"
    elif k < 0.21:
        return "Slight (0.00–0.20)"
    elif k < 0.41:
        return "Fair (0.21–0.40)"
    elif k < 0.61:
        return "Moderate (0.41–0.60)"
    elif k < 0.81:
        return "Substantial (0.61–0.80)"
    else:
        return "Almost perfect (0.81–1.00)"


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python qualitative.py <ratings.csv> [--json]")
        print("\nCodebook codes:")
        for code, info in CODEBOOK.items():
            print(f"  {code}: {info['label']}")
        sys.exit(0)
    output_json = "--json" in sys.argv
    report = compare_raters(sys.argv[1], output_json=output_json)
    if not output_json:
        print(f"\nInter-Rater Reliability Report")
        print(f"  Items:   {report['n_items']}")
        print(f"  Raters:  {report['n_raters']}")
        for pair, stats in report["pairwise"].items():
            print(f"\n  {pair}")
            print(f"    κ = {stats['kappa']}  95% CI [{stats['ci_lower']}, {stats['ci_upper']}]")
            print(f"    {stats['interpretation']}")
        if report["fleiss_kappa"]:
            fk = report["fleiss_kappa"]
            print(f"\n  Fleiss' κ (all raters) = {fk['kappa']}  {fk['interpretation']}")
        print()
