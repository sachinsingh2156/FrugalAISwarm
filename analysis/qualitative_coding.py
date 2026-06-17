"""
Qualitative trace coding helper (RQ1 analysis).

Reads agent traces from ChromaDB and presents them for coding.
Coding scheme is based on the role-formation patterns the synopsis targets.

Codes:
  SP  — Specialisation pattern  (agent consistently picks one task type)
  DP  — Diversification pattern (agent covers multiple task types)
  LF  — Leadership / high-Shapley (agent with persistently high contribution score)
  CF  — Conformity failure (agent changes answer when peer output differs)
  AN  — Anomaly detected (embedding drift flagged)
  NE  — No evidence of specialisation

Usage:
    python analysis/qualitative_coding.py --run-id <run_id>
    python analysis/qualitative_coding.py --interactive   # manual coding UI
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

from frugal_swarm.memory.chroma_store import ChromaStore

console = Console()

CODES = {
    "SP": "Specialisation pattern",
    "DP": "Diversification pattern",
    "LF": "Leadership / high-Shapley",
    "CF": "Conformity failure",
    "AN": "Anomaly detected",
    "NE": "No evidence of specialisation",
}


def auto_code(traces: list[dict]) -> dict[str, str]:
    """
    Auto-assign preliminary codes based on metadata patterns.
    Returns {doc_id: code}.
    """
    # Group by agent
    from collections import Counter, defaultdict
    agent_families: dict[str, list[str]] = defaultdict(list)
    doc_to_agent: dict[str, str] = {}

    for trace in traces:
        meta = trace["metadata"]
        doc = trace["document"]
        agent_id = meta.get("agent_id", "unknown")
        family = meta.get("task_family", "unknown")
        agent_families[agent_id].append(family)
        doc_to_agent[meta.get("doc_id", "")] = agent_id

    # Compute specialisation index per agent
    agent_specialisation: dict[str, str] = {}
    for agent_id, families in agent_families.items():
        if not families:
            agent_specialisation[agent_id] = "NE"
            continue
        counts = Counter(families)
        modal_frac = counts.most_common(1)[0][1] / len(families)
        if modal_frac > 0.7:
            agent_specialisation[agent_id] = "SP"
        elif modal_frac < 0.4:
            agent_specialisation[agent_id] = "DP"
        else:
            agent_specialisation[agent_id] = "NE"

    # Map back to traces (doc_ids not stored in metadata in this version)
    # Simple approach: code each trace by its agent's pattern
    codes: dict[str, str] = {}
    for i, trace in enumerate(traces):
        meta = trace["metadata"]
        agent_id = meta.get("agent_id", "unknown")
        code = agent_specialisation.get(agent_id, "NE")
        codes[f"trace_{i}"] = code
    return codes


def print_trace(trace: dict, idx: int) -> None:
    meta = trace["metadata"]
    doc = trace["document"]
    console.rule(f"[bold]Trace {idx+1}[/bold]")
    console.print(f"Agent: [cyan]{meta.get('agent_id')}[/cyan] | Family: [yellow]{meta.get('task_family')}[/yellow] | Round: {meta.get('round_num')}")
    console.print(f"Tokens: {meta.get('tokens_used')} | Uncertainty: {meta.get('uncertainty', 'N/A'):.3f}" if meta.get('uncertainty', -1) >= 0 else f"Tokens: {meta.get('tokens_used')} | Uncertainty: N/A")
    console.print(f"[dim]Prompt:[/dim] {doc.get('prompt', '')[:120]}…")
    console.print(f"[dim]Response:[/dim] {doc.get('response', '')[:200]}…")
    if doc.get("upstream_context"):
        console.print(f"[dim]Upstream:[/dim] {len(doc['upstream_context'])} inputs")


def interactive_coding(traces: list[dict], chroma: ChromaStore) -> None:
    """Present each trace and ask for a code."""
    console.print(f"\nCoding scheme: {json.dumps(CODES, indent=2)}\n")
    for i, trace in enumerate(traces):
        print_trace(trace, i)
        code = Prompt.ask(
            f"Code (SP/DP/LF/CF/AN/NE) [{i+1}/{len(traces)}]",
            choices=list(CODES.keys()),
            default="NE",
        )
        # Update in ChromaDB (best-effort; doc_id not easily available without storing it)
        console.print(f"  → Coded as [green]{code}[/green]: {CODES[code]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-traces", default=100, type=int)
    args = parser.parse_args()

    chroma = ChromaStore()
    traces = chroma.query_traces(
        run_id=args.run_id,
        agent_id=args.agent_id,
        task_family=args.family,
        n_results=args.max_traces,
    )
    console.print(f"[bold]Loaded {len(traces)} traces[/bold]")

    if args.interactive:
        interactive_coding(traces, chroma)
    else:
        auto_codes = auto_code(traces)
        table = Table(title=f"Auto-coded traces (run={args.run_id})")
        table.add_column("Index")
        table.add_column("Agent")
        table.add_column("Family")
        table.add_column("Auto-code")
        for i, trace in enumerate(traces):
            meta = trace["metadata"]
            table.add_row(
                str(i+1),
                meta.get("agent_id", "?"),
                meta.get("task_family", "?"),
                auto_codes.get(f"trace_{i}", "?"),
            )
        console.print(table)
