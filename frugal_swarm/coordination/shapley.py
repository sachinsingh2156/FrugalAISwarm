"""
Shapley-based contribution estimator (SELFORG method, Tastan et al. 2026).

Per-agent Shapley score = cosine similarity between the agent's response
embedding and the mean embedding of the agent pool, normalised within a
sliding window so scores sum to 1.

This file is stateless; the caller passes embeddings in and receives scores out.
Rolling scores are stored in ChromaDB (memory layer) and read back by the DAG
builder.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from frugal_swarm.model.embedder import cosine_similarity, mean_embedding, shapley_scores


class ShapleyEstimator:
    """
    Maintains a sliding-window history of raw Shapley scores per agent and
    exposes rolling normalised values for the DAG builder.

    window_size: number of past rounds included in the rolling mean.
    """

    def __init__(self, agent_ids: list[str], window_size: int = 10) -> None:
        self.agent_ids = agent_ids
        self.window_size = window_size
        # agent_id → deque of raw scores
        self._history: dict[str, deque[float]] = {
            aid: deque(maxlen=window_size) for aid in agent_ids
        }

    def update(
        self, embeddings: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """
        Compute per-agent Shapley scores for the current round from their
        response embeddings.  Append to history and return the round scores.

        embeddings: {agent_id: embedding_vector}
        Returns: {agent_id: normalised_score_this_round}
        """
        ids = list(embeddings.keys())
        vecs = [embeddings[aid] for aid in ids]
        scores = shapley_scores(vecs)
        result: dict[str, float] = {}
        for aid, score in zip(ids, scores):
            self._history[aid].append(score)
            result[aid] = score

        # Agents that did not participate get a tiny score to avoid zero
        for aid in self.agent_ids:
            if aid not in result:
                self._history[aid].append(1e-6)
                result[aid] = 1e-6

        return result

    def rolling_scores(self) -> dict[str, float]:
        """
        Rolling mean Shapley scores across the sliding window.
        Normalised so they sum to 1.
        """
        means = {
            aid: float(np.mean(list(hist))) if hist else 1e-6
            for aid, hist in self._history.items()
        }
        total = sum(means.values()) or 1.0
        return {aid: v / total for aid, v in means.items()}
