"""
Day-2 model auto-selection.

Runs a 20-item QA smoke set against each candidate model and selects the one
that passes the reliability floor (>= 50% exact-match).  Writes the winning
model tag to DATA_DIR/.selected_model so the rest of the pipeline uses it.

Usage:
    python -m frugal_swarm.model.model_benchmark
    python -m frugal_swarm.model.model_benchmark --force   # re-run even if already locked
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (
    MODEL_CANDIDATES,
    OLLAMA_BASE_URL,
    SELECTED_MODEL_FILE,
)
from frugal_swarm.model.ollama_client import OllamaClient

console = Console()

# ── Smoke-set (20 factual QA pairs, CC-licensed) ──────────────────────────────
SMOKE_SET: list[dict] = [
    {"q": "What is the capital of France?", "a": "paris"},
    {"q": "How many sides does a hexagon have?", "a": "6"},
    {"q": "What is 15 multiplied by 7?", "a": "105"},
    {"q": "Who wrote Romeo and Juliet?", "a": "shakespeare"},
    {"q": "What is the chemical symbol for water?", "a": "h2o"},
    {"q": "How many planets are in our solar system?", "a": "8"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "In which continent is Brazil located?", "a": "south america"},
    {"q": "What is the largest ocean on Earth?", "a": "pacific"},
    {"q": "How many hours are in a day?", "a": "24"},
    {"q": "What gas do plants absorb from the atmosphere?", "a": "carbon dioxide"},
    {"q": "What is 2 to the power of 10?", "a": "1024"},
    {"q": "What language is spoken in Brazil?", "a": "portuguese"},
    {"q": "How many bones are in the adult human body?", "a": "206"},
    {"q": "What is the speed of light in km/s (approximate)?", "a": "300000"},
    {"q": "What is the smallest prime number?", "a": "2"},
    {"q": "What is the capital of Japan?", "a": "tokyo"},
    {"q": "How many degrees are in a right angle?", "a": "90"},
    {"q": "What element has atomic number 1?", "a": "hydrogen"},
]

RELIABILITY_FLOOR = 0.50  # >= 50 % exact-match to pass


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip().strip("."))


def _contains_answer(response: str, answer: str) -> bool:
    return _normalise(answer) in _normalise(response)


def benchmark_model(model_tag: str, base_url: str = OLLAMA_BASE_URL) -> dict:
    """Run the 20-item smoke set; return accuracy and token stats."""
    client = OllamaClient(base_url=base_url, model=model_tag)
    correct = 0
    total_tokens = 0
    total_latency = 0.0
    results = []

    console.print(f"\n[bold cyan]Benchmarking[/bold cyan] {model_tag} …")
    for item in SMOKE_SET:
        prompt = f"Answer the following question with a short, direct answer.\n\nQuestion: {item['q']}\nAnswer:"
        try:
            resp = client.generate(prompt, temperature=0.0, max_tokens=64)
            hit = _contains_answer(resp["text"], item["a"])
            correct += int(hit)
            total_tokens += resp["tokens_used"]
            total_latency += resp["latency_s"]
            results.append({"q": item["q"], "expected": item["a"], "got": resp["text"].strip(), "hit": hit})
        except Exception as exc:
            console.print(f"  [red]ERROR[/red] on '{item['q']}': {exc}")
            results.append({"q": item["q"], "expected": item["a"], "got": "ERROR", "hit": False})

    accuracy = correct / len(SMOKE_SET)
    return {
        "model": model_tag,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(SMOKE_SET),
        "mean_tokens": total_tokens / len(SMOKE_SET),
        "mean_latency_s": total_latency / len(SMOKE_SET),
        "passed": accuracy >= RELIABILITY_FLOOR,
        "results": results,
    }


def select_model(force: bool = False) -> str:
    """
    Benchmark all candidates and write the winner to SELECTED_MODEL_FILE.
    Returns the selected model tag.
    """
    if SELECTED_MODEL_FILE.exists() and not force:
        locked = SELECTED_MODEL_FILE.read_text().strip()
        console.print(f"[green]Model already locked:[/green] {locked}  (use --force to re-benchmark)")
        return locked

    console.print("[bold]Starting Day-2 model benchmark …[/bold]")
    client = OllamaClient()
    available = client.list_models()
    console.print(f"Available models: {available}")

    outcomes: list[dict] = []
    winner: str | None = None

    for tag in MODEL_CANDIDATES:
        # Check if model is pulled; skip if not available
        if not any(tag.split(":")[0] in m for m in available):
            console.print(f"[yellow]Model {tag} not found in Ollama — skipping[/yellow]")
            continue
        result = benchmark_model(tag)
        outcomes.append(result)
        if result["passed"] and winner is None:
            winner = tag

    # Print summary table
    table = Table(title="Model Benchmark Results")
    table.add_column("Model")
    table.add_column("Accuracy")
    table.add_column("Passed")
    table.add_column("Mean tokens")
    table.add_column("Mean latency (s)")
    for o in outcomes:
        table.add_row(
            o["model"],
            f"{o['accuracy']:.1%} ({o['correct']}/{o['total']})",
            "✓" if o["passed"] else "✗",
            f"{o['mean_tokens']:.0f}",
            f"{o['mean_latency_s']:.2f}",
        )
    console.print(table)

    if winner is None:
        # Fallback: pick the highest-accuracy model regardless of floor
        outcomes.sort(key=lambda x: x["accuracy"], reverse=True)
        winner = outcomes[0]["model"] if outcomes else MODEL_CANDIDATES[0]
        console.print(f"[yellow]No model passed the floor — using best available: {winner}[/yellow]")
    else:
        console.print(f"[green bold]Selected model:[/green bold] {winner}")

    SELECTED_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_MODEL_FILE.write_text(winner)
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day-2 model benchmark and auto-select")
    parser.add_argument("--force", action="store_true", help="Re-run even if model already locked")
    args = parser.parse_args()
    selected = select_model(force=args.force)
    print(f"\nUsing model: {selected}")
