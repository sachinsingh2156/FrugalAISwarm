# ── Frugal AI Agent Swarm — Dashboard Container ───────────────────────────────
# Base: Python 3.11 slim (no GPU; CPU inference via Ollama sidecar)
FROM python:3.11-slim

LABEL maintainer="Sachin Singh <claude.sachin1@gmail.com>"
LABEL description="Frugal AI Agent Swarm — Phase-1 Pilot Dashboard"

# System deps (no build tools needed for CPU-only psutil wheel)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy project
COPY . .

# Data directory (mounted as volume in docker-compose)
RUN mkdir -p /app/data

EXPOSE 5050

# Health-check — verifies Flask is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:5050/api/status || exit 1

CMD ["python", "dashboard_server.py"]
