"""
Dashboard server for the Frugal AI Agent Swarm.

Serves the HTML dashboard and streams swarm execution events via SSE.

Usage:
    python dashboard_server.py
    -> open http://localhost:5050

Endpoints:
    GET  /                   -> dashboard.html
    POST /api/run            -> SSE stream of swarm events
    GET  /api/status         -> server + model status
    GET  /api/corpus         -> available tasks for the Live Run task picker
    GET  /api/results        -> Result1 + Result2 run_log.jsonl data
    GET  /api/corpus_tasks   -> task prompts+references for family/seed (Results viewer)

Config names accepted by POST /api/run:
    "C1"  — single-agent baseline
    "A1"  — fixed-role, single model
    "A2"  — fixed-role, multi-model
    "A3"  — self-organising, single model  (default)
    "A4"  — self-organising, multi-model
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from queue import Queue

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))

from config import VerificationMode, get_model, MODEL_POOL
from frugal_swarm.corpus.loader import load_family
from frugal_swarm.coordination.state import Task
from frugal_swarm.dashboard.runner import run_c1, run_a1, run_a2, run_a3, run_a4
from frugal_swarm.model.ollama_client import OllamaClient

app = Flask(__name__, static_folder=".")
CORS(app)

DASHBOARD_DIR = os.path.dirname(__file__)


@app.route("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "dashboard.html")


@app.route("/api/status")
def status():
    client = OllamaClient()
    available = client.list_models() if client.is_available() else []
    return jsonify({
        "ollama_available": client.is_available(),
        "model": get_model(),
        "available_models": available,
        "model_pool": MODEL_POOL,
    })


@app.route("/api/corpus")
def corpus():
    family = request.args.get("family", "formative_assessment_drafting")
    try:
        tasks = load_family(family, seed=42)
        return jsonify([{
            "task_id": t.task_id,
            "prompt": t.prompt,
            "family": t.family,
            "reference": t.reference,
        } for t in tasks[:20]])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/results")
def api_results():
    """Return structured per-cell data from both Result1/ and Result2/ run_log.jsonl files."""
    def load_log(path, run_label):
        cells = []
        if not os.path.exists(path):
            return cells
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    d["run"] = run_label
                    cells.append(d)
                except Exception:
                    pass
        return cells

    def add_overhead(cells):
        c1 = [c for c in cells if c.get("config") == "C1"]
        baseline = sum(c["energy_kwh"] for c in c1) / len(c1) if c1 else 0
        for c in cells:
            c["energy_overhead"] = round(c["energy_kwh"] / baseline, 2) if baseline > 0 else 0
        return cells

    r1 = add_overhead(load_log(os.path.join(DASHBOARD_DIR, "Result1", "run_log.jsonl"), "R1"))
    r2 = add_overhead(load_log(os.path.join(DASHBOARD_DIR, "Result2", "run_log.jsonl"), "R2"))

    return jsonify({
        "result1": r1,
        "result2": r2,
        "has_result1": len(r1) > 0,
        "has_result2": len(r2) > 0,
    })


@app.route("/api/corpus_tasks")
def api_corpus_tasks():
    """Return task prompts + reference answers for a given family/seed (for Results input viewer)."""
    family = request.args.get("family", "formative_assessment_drafting")
    seed   = request.args.get("seed", None)
    n      = int(request.args.get("n", 10))
    try:
        seed_int = int(seed) if seed is not None else None
        tasks = load_family(family, seed=seed_int)
        return jsonify([{
            "task_id":   t.task_id,
            "prompt":    t.prompt,
            "family":    t.family,
            "reference": t.reference,
        } for t in tasks[:n]])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/task_outputs")
def api_task_outputs():
    """Return per-task outputs from task_outputs.jsonl (written by future experiment runs)."""
    run     = request.args.get("run", "R2")   # R1 or R2
    config  = request.args.get("config", None)
    family  = request.args.get("family", None)
    seed    = request.args.get("seed", None)

    subdir  = "Result1" if run == "R1" else "Result2"
    path    = os.path.join(DASHBOARD_DIR, subdir, "task_outputs.jsonl")

    if not os.path.exists(path):
        return jsonify({"records": [], "available": False})

    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if config and d.get("config") != config:
                        continue
                    if family and d.get("family") != family:
                        continue
                    if seed is not None and str(d.get("seed")) != str(seed):
                        continue
                    records.append(d)
                except Exception:
                    pass
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"records": records, "available": True})


@app.route("/api/run", methods=["POST"])
def run_task():
    body = request.json or {}
    task_prompt = body.get("prompt", "").strip()
    family      = body.get("family", "formative_assessment_drafting")
    config_name = body.get("config", "A3")   # C1 | A1 | A2 | A3 | A4
    mode_str    = body.get("mode", "none")   # full | selective | none
    swarm_size  = int(body.get("swarm_size", 3))

    if not task_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    task = Task(
        task_id="dashboard_task",
        family=family,
        prompt=task_prompt,
        reference=body.get("reference", ""),
        priority=1,
    )
    agent_ids = [f"agent_{i}" for i in range(swarm_size)]
    mode = VerificationMode(mode_str)

    event_queue: Queue = Queue()

    def worker():
        try:
            client = OllamaClient()
            if config_name == "C1":
                gen = run_c1(task, agent_ids[:1], client)
            elif config_name == "A1":
                gen = run_a1(task, agent_ids, client)
            elif config_name == "A2":
                gen = run_a2(task, agent_ids, client, model_assignment=MODEL_POOL)
            elif config_name == "A4":
                gen = run_a4(task, agent_ids, client,
                             model_assignment=MODEL_POOL, mode=mode)
            else:
                gen = run_a3(task, agent_ids, client, mode=mode)

            for event in gen:
                event_queue.put(event)

        except Exception:
            event_queue.put({"type": "error", "message": traceback.format_exc()})
        finally:
            event_queue.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def sse_stream():
        yield "retry: 3000\n\n"
        while True:
            event = event_queue.get()
            if event is None:
                yield "event: done\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        sse_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    print("\n  Frugal AI Agent Swarm — Dashboard")
    print("   Open: http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
