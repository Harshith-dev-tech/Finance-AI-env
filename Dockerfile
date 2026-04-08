# ─── Personal Finance RL Environment — Dockerfile ───────────────────────────
# Base: slim Python 3.11 image for a small footprint
FROM python:3.11-slim

# Metadata
LABEL name="personal-finance-control" \
      version="1.0.0" \
      description="OpenEnv personal finance RL environment"

# Set working directory
WORKDIR /app

# Copy dependency list first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY env.py tasks.py inference.py server.py openenv.yaml ./

# Default command: run the evaluation script
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
