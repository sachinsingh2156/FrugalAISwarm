# ── Frugal AI Agent Swarm — Makefile ─────────────────────────────────────────
.PHONY: help up down logs pull-models check-ollama install run test clean

SHELL := /bin/bash
COMPOSE = docker compose

help:
	@echo ""
	@echo "  Frugal AI Agent Swarm — available targets:"
	@echo ""
	@echo "  make check-ollama   Verify local Ollama is running"
	@echo "  make pull-models    Pull required models into local Ollama"
	@echo "  make run            Run dashboard locally (recommended)"
	@echo "  make up             Start dashboard in Docker (Ollama stays on host)"
	@echo "  make up-d           Start dashboard detached"
	@echo "  make down           Stop dashboard container"
	@echo "  make logs           Tail dashboard logs"
	@echo "  make install        Install Python deps locally"
	@echo "  make test           Run smoke tests"
	@echo "  make clean          Remove dashboard container and volumes"
	@echo ""

check-ollama:
	@curl -sf http://localhost:11434/api/tags >/dev/null \
	  && echo "Ollama is running at http://localhost:11434" \
	  || (echo "Ollama not reachable. Start it with: ollama serve" && exit 1)

pull-models: check-ollama
	@echo "Pulling Ollama models locally (1B–4B, frugal edge inference, this may take a few minutes)..."
	ollama pull qwen2.5:1.5b
	ollama pull llama3.2:3b
	ollama pull phi3:mini
	@echo "Done. Models ready in local Ollama (qwen2.5:1.5b, llama3.2:3b, phi3:mini)."

up:
	$(COMPOSE) up

up-d: check-ollama
	$(COMPOSE) up -d
	@echo "Dashboard: http://localhost:5050  (Ollama: http://localhost:11434 on host)"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f dashboard

install:
	pip install -r requirements.txt --break-system-packages

run: check-ollama
	python3 dashboard_server.py

test:
	python -c "import ast, pathlib; \
	  files = list(pathlib.Path('.').rglob('*.py')); \
	  [ast.parse(f.read_text()) for f in files]; \
	  print(f'Syntax OK: {len(files)} files')"

clean:
	$(COMPOSE) down -v
	@echo "Dashboard container and volumes removed."
