"""
ChromaDB store for the Frugal AI Agent Swarm.

Three collections:
  - agent_traces    : every agent action with full metadata + embedding
  - shapley_scores  : per-round rolling Shapley scores per agent
  - task_definitions: the task corpus (for retrieval-augmented generation later)

The trace collection is what the qualitative-coding workflow reads from.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_TRACES_COLLECTION,
    CHROMA_SCORES_COLLECTION,
    CHROMA_TASKS_COLLECTION,
)
from frugal_swarm.coordination.state import Task


class ChromaStore:
    """Wrapper around chromadb.PersistentClient with domain-specific methods."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._traces = self._client.get_or_create_collection(
            CHROMA_TRACES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._scores = self._client.get_or_create_collection(
            CHROMA_SCORES_COLLECTION,
        )
        self._tasks = self._client.get_or_create_collection(
            CHROMA_TASKS_COLLECTION,
        )

    # ── Agent trace logging ───────────────────────────────────────────────────

    def log_agent_action(
        self,
        run_id: str,
        agent_id: str,
        task: Task,
        response: str,
        embedding: list[float],
        tokens_used: int,
        latency_s: float,
        uncertainty: float | None,
        upstream: list[str],
        round_num: int,
        qualitative_code: str = "",
    ) -> str:
        """Insert one agent-action trace.  Returns the trace document ID."""
        doc_id = f"{run_id}_{agent_id}_{task.task_id}_{uuid.uuid4().hex[:8]}"
        metadata = {
            "run_id": run_id,
            "agent_id": agent_id,
            "task_id": task.task_id,
            "task_family": task.family,
            "tokens_used": tokens_used,
            "latency_s": latency_s,
            "uncertainty": uncertainty if uncertainty is not None else -1.0,
            "had_upstream": int(bool(upstream)),
            "round_num": round_num,
            "qualitative_code": qualitative_code,
            "timestamp": time.time(),
        }
        document = json.dumps({
            "prompt": task.prompt,
            "response": response,
            "upstream_context": upstream,
        })
        self._traces.add(
            ids=[doc_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        return doc_id

    def query_traces(
        self,
        run_id: str | None = None,
        agent_id: str | None = None,
        task_family: str | None = None,
        n_results: int = 100,
    ) -> list[dict]:
        """Query traces with optional filters."""
        where: dict[str, Any] = {}
        if run_id:
            where["run_id"] = run_id
        if agent_id:
            where["agent_id"] = agent_id
        if task_family:
            where["task_family"] = task_family

        result = self._traces.get(
            where=where if where else None,
            include=["documents", "metadatas"],
        )
        traces = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            traces.append({"document": json.loads(doc), "metadata": meta})
        return traces

    def count_traces(self, run_id: str | None = None) -> int:
        if run_id:
            return self._traces.get(where={"run_id": run_id}, include=[])["ids"].__len__()
        return self._traces.count()

    # ── Shapley score logging ─────────────────────────────────────────────────

    def log_shapley_scores(
        self,
        run_id: str,
        round_num: int,
        scores: dict[str, float],
    ) -> None:
        """Log per-agent Shapley scores for one round."""
        doc_id = f"{run_id}_round_{round_num}"
        self._scores.upsert(
            ids=[doc_id],
            documents=[json.dumps(scores)],
            metadatas=[{"run_id": run_id, "round_num": round_num, "timestamp": time.time()}],
        )

    def get_shapley_history(self, run_id: str) -> list[dict]:
        result = self._scores.get(where={"run_id": run_id}, include=["documents", "metadatas"])
        history = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            history.append({"round": meta["round_num"], "scores": json.loads(doc)})
        return sorted(history, key=lambda x: x["round"])

    # ── Task corpus storage ───────────────────────────────────────────────────

    def store_tasks(self, tasks: list[Task]) -> None:
        """Upsert the task corpus into ChromaDB for later retrieval."""
        for task in tasks:
            self._tasks.upsert(
                ids=[task.task_id],
                documents=[task.prompt],
                metadatas=[{
                    "task_id": task.task_id,
                    "family": task.family,
                    "reference": task.reference,
                    "priority": task.priority,
                }],
            )

    # ── Qualitative code update ───────────────────────────────────────────────

    def update_qualitative_code(self, doc_id: str, code: str) -> None:
        """Set the qualitative code on an existing trace (for RQ1 coding)."""
        self._traces.update(ids=[doc_id], metadatas=[{"qualitative_code": code}])
