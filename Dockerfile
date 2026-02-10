FROM python:3.12-slim-bookworm

# Real-time log output + skip .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dep for scikit-learn OpenMP parallelism
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching: deps change rarely, code changes often)
COPY pyproject.toml requirements.txt ./

# Install pinned dependencies, then the optional groups (anthropic + pytest)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "anthropic>=0.40.0" "pytest>=7.0"

# Copy source code and scripts
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install the package so persona_clustering is importable
RUN pip install --no-cache-dir -e ".[all]"

CMD ["python", "scripts/run_pipeline.py", "--help"]
