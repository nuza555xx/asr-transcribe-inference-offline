FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

# Install Python 3.12, pip, curl, and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Symlink python3.12 to python & upgrade pip
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    python -m pip install --upgrade pip

# Install Poetry
ENV POETRY_VERSION=1.8.4 \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python -
# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy Poetry config\ COPY pyproject.toml poetry.lock* ./

# Install dependencies (no dev packages)
RUN poetry install --no-interaction --no-ansi

# Copy application code
COPY src ./src

# Expose application port
EXPOSE 8000

# Optimize vLLM to use precompiled kernels
ENV VLLM_USE_PRECOMPILED=1

# Default command: launch uvicorn
CMD ["poetry", "run", "uvicorn", "src.transcribe_vllm.main:app", "--host", "0.0.0.0", "--port", "8000"]
