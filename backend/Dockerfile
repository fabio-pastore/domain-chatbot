ARG CUDA_IMAGE=runtime
FROM nvidia/cuda:12.4.1-cudnn-${CUDA_IMAGE}-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    gcc \
    g++ \
    libgomp1 \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Pre-download the models weights (~150MB + ~100MB)
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Conditionally rebuild llama-cpp-python with CUDA support
ARG LLAMA_CUDA=OFF
RUN if [ "$LLAMA_CUDA" = "ON" ]; then \
        if ! command -v nvcc >/dev/null 2>&1; then \
            echo "ERROR: LLAMA_CUDA=ON requires CUDA_IMAGE=devel" && exit 1; \
        fi; \
        apt-get update && apt-get install -y --no-install-recommends cmake && \
        CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir && \
        apt-get purge -y cmake && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*; \
    fi

# Purge build tools only AFTER the optional CUDA rebuild
RUN apt-mark manual libmariadb3 libgomp1 2>/dev/null; \
    apt-get purge -y gcc g++ python3.12-dev software-properties-common && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8001

CMD ["uvicorn", "src.api_server.server:app", "--host", "0.0.0.0", "--port", "8001"]