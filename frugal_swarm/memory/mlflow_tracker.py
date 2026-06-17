"""
MLflow experiment tracker for the Frugal AI Agent Swarm.

Logs experiment-level metrics for each of the 81 evaluation cells:
  - per-task success, total tokens, wall-clock time
  - H1–H3 derived quantities

One MLflow run corresponds to one experimental cell (config × family × seed × mode).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import mlflow

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


class MLflowTracker:
    def __init__(
        self,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run: mlflow.ActiveRun | None = None
        self._start_time: float = 0.0

    @contextmanager
    def run(
        self,
        run_name: str,
        params: dict[str, Any] | None = None,
    ) -> Iterator["MLflowTracker"]:
        """Context manager for one evaluation cell."""
        with mlflow.start_run(run_name=run_name) as active_run:
            self._run = active_run
            self._start_time = time.time()
            if params:
                mlflow.log_params(params)
            yield self
            mlflow.log_metric("wall_clock_s", time.time() - self._start_time)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if step is not None:
            mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metric(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact_path(self, path: str) -> None:
        mlflow.log_artifact(path)

    def log_hypothesis_results(
        self,
        h1_stability: float,
        h1_passed: bool,
        h2_token_reduction: float | None,
        h2_reliability_retention: float | None,
        h2_passed: bool | None,
        h3_efficiency_ratio: float | None,
        h3_passed: bool | None,
    ) -> None:
        """Log the H1–H3 hypothesis verdicts as MLflow metrics."""
        mlflow.log_metrics({
            "h1_role_stability": h1_stability,
            "h1_passed": int(h1_passed),
        })
        if h2_token_reduction is not None:
            mlflow.log_metrics({
                "h2_token_reduction": h2_token_reduction,
                "h2_reliability_retention": h2_reliability_retention or 0.0,
                "h2_passed": int(h2_passed or False),
            })
        if h3_efficiency_ratio is not None:
            mlflow.log_metrics({
                "h3_efficiency_ratio": h3_efficiency_ratio,
                "h3_passed": int(h3_passed or False),
            })


class RunLogger:
    """
    Lightweight logger passed to the swarm graph.
    Bridges between the graph (real-time) and ChromaDB + MLflow (async).
    """

    def __init__(
        self,
        chroma_store: Any,
        mlflow_tracker: MLflowTracker | None = None,
    ) -> None:
        self.chroma = chroma_store
        self.mlflow = mlflow_tracker
        self._round_metrics: list[dict] = []

    def log_agent_action(self, **kwargs: Any) -> None:
        self.chroma.log_agent_action(**kwargs)

    def log_round_summary(
        self,
        run_id: str,
        round_num: int,
        shapley_scores: dict[str, float],
        rolling_scores: dict[str, float],
        dag: dict[str, list[str]],
        board: dict[str, int],
        role_summary: dict[str, Any],
    ) -> None:
        self.chroma.log_shapley_scores(run_id, round_num, rolling_scores)
        record = {
            "round_num": round_num,
            "shapley_scores": shapley_scores,
            "rolling_scores": rolling_scores,
            "dag_edge_count": sum(len(v) for v in dag.values()),
            "board": board,
            "mean_role_stability": role_summary.get("mean_role_stability", 0.0),
            "specialisation_entropy": role_summary.get("specialisation_entropy", 0.0),
        }
        self._round_metrics.append(record)
        if self.mlflow:
            self.mlflow.log_metrics(
                {
                    "dag_edges": record["dag_edge_count"],
                    "mean_role_stability": record["mean_role_stability"],
                    "specialisation_entropy": record["specialisation_entropy"],
                    "tasks_done": board.get("done", 0),
                },
                step=round_num,
            )
