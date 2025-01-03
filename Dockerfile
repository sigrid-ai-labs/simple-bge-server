FROM nvidia/cuda:12.1.0-base-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=60 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv and create virtual environment
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv venv .venv

# Activate virtual environment in subsequent commands
ENV PATH="/root/.local/bin:/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Copy source code and requirements
COPY main.py .
COPY pyproject.toml .

# Install dependencies using uv
RUN uv pip install \
    "fastapi>=0.115.6" \
    "flagembedding>=1.3.3" \
    "pydantic>=2.10.4" \
    "astral-uv>=1.0.2" \
    "torch>=2.2.0" \
    "numpy>=1.24.0"

# Start second stage
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

WORKDIR /app

# Copy virtual environment and uv from base stage
COPY --from=base /app/.venv /app/.venv
COPY --from=base /root/.local/bin/uv /usr/local/bin/uv
COPY --from=base /app/main.py /app/main.py

# Set the environment variables for the new stage
ENV PATH="/usr/local/bin:/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Expose port
EXPOSE 8000

# Start the application with Astral
# CMD ["python", "-m", "astral", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=80", "--loop=uvloop"]
