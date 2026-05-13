ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install base runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Conditionally install Python 3.12 for base images that don't have it (e.g., CUDA images)
RUN if ! command -v python3.12 >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa -y && \
        apt-get update && apt-get install -y --no-install-recommends \
            python3.12 \
            python3.12-dev \
            python3.12-venv && \
        curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
        update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

ARG REQUIREMENTS_FILE=requirements-cpu.txt
COPY ${REQUIREMENTS_FILE} requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    CMAKE_BUILD_PARALLEL_LEVEL=4 MAX_JOBS=4 pip install -r requirements.txt

# Force uninstall broken PyTorch versions and sentence-transformers, force the PyTorch index, THEN install sentence-transformers (only way to fix the CUDA 13 issue)
RUN pip uninstall -y torch torchvision torchaudio sentence-transformers && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir sentence-transformers

# Pre-download the models weights (~150MB + ~100MB)
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Purge build tools
RUN apt-mark manual libmariadb3 libgomp1 2>/dev/null; \
    apt-get purge -y gcc g++ 2>/dev/null || true; \
    apt-get purge -y cmake 2>/dev/null || true; \
    apt-get purge -y python3.12-dev 2>/dev/null || true; \
    apt-get purge -y software-properties-common 2>/dev/null || true; \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8001

CMD ["uvicorn", "src.api_server.server:app", "--host", "0.0.0.0", "--port", "8001"]
