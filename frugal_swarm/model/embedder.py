"""
Sentence embedding wrapper using all-MiniLM-L6-v2.

Used by the Shapley contribution estimator and the role-stability monitor.
The model is lazy-loaded and cached as a module-level singleton.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(texts: list[str] | str) -> np.ndarray:
    """
    Embed one or more texts.
    Returns an ndarray of shape (N, D) for a list, or (D,) for a single string.
    """
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    vecs = _get_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs[0] if single else vecs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two normalised unit vectors.
    Both inputs should already be L2-normalised (as returned by embed()).
    """
    return float(np.dot(a, b))


def mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """Compute the mean of a list of embedding vectors (then re-normalise)."""
    mat = np.stack(embeddings, axis=0)
    mu = mat.mean(axis=0)
    norm = np.linalg.norm(mu)
    return mu / norm if norm > 1e-9 else mu


def shapley_scores(embeddings: list[np.ndarray]) -> list[float]:
    """
    Approximate Shapley contribution scores (SELFORG method):
    cos-sim of each agent embedding vs. the mean embedding of the pool,
    then normalise so scores sum to 1.

    Args:
        embeddings: list of per-agent response embeddings (L2-normalised).
    Returns:
        list of floats in [0, 1] summing to 1.
    """
    if not embeddings:
        return []
    mu = mean_embedding(embeddings)
    raw = [cosine_similarity(e, mu) for e in embeddings]
    # Shift to non-negative before normalising
    min_raw = min(raw)
    shifted = [r - min_raw + 1e-9 for r in raw]
    total = sum(shifted)
    return [s / total for s in shifted]
