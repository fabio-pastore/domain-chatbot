# Sapienza-DC
<div>
  <a href="https://github.com/fabio-pastore">
    <img src="https://github.com/fabio-pastore.png" width="40" height="40" style="border-radius:50%" alt="Fabio Pastore"/>
  </a>
  &nbsp;
  <a href="https://github.com/ywu24">
    <img src="https://github.com/ywu24.png" width="40" height="40" style="border-radius:50%" alt="Yi Hao Wu"/>
  </a>
  &nbsp;
  <a href="https://github.com/Zhanytrix">
    <img src="https://github.com/Zhanytrix.png" width="40" height="40" style="border-radius:50%" alt="Alessandro Zannone"/>
  </a>
  <br/>
  <sub>Built by <b>Fabio Pastore</b> · <b>Yi Hao Wu</b> · <b>Alessandro Zannone</b></sub>
</div>

<br>
<p align="center">
  <img width="200" height="200" alt="sdc-logo" src="https://github.com/user-attachments/assets/ed3cf0ed-c4ef-4c29-bca9-3b35331ea00d" />
</p>

A domain-aware retrieval-augmented generation chatbot powered by a local large language model. Sapienza-DC answers user queries by searching across curated domains, extracting relevant content through web parsing, and generating responses with explicit reliability scoring and source attribution.

## Objectives

Sapienza-DC was built to explore the feasibility of a fully local, domain-constrained RAG system. The project targets environments with limited computational resources (consumer CPUs, 16 GB RAM, optional GPU) while maintaining a production-quality user experience. It serves as a testbed for:

- **Structured prompt engineering** for query validation, routing, and grounded generation
- **Efficient embedding** and reranking strategies for small models
- Integration of external web parsing tools (_MWP_) into a **coherent RAG pipeline**
- **Transparent** AI through explicit *reliability scoring and source attribution*

## Running the project

### Prerequisites

#### System Requirements
- Docker with BuildKit enabled
  - Enable BuildKit by setting `DOCKER_BUILDKIT=1` or using `docker buildx`
- RAM: Minimum 8 GB (16 GB+ recommended for Windows 11 or complex models)
- Storage: At least 20 GB free disk space (for models + dependencies)
- CPU: Recommend Ryzen 5 4600H or better. Total generation time from tests with the 4600H is ~2min 15s.
- Optional:
  - NVIDIA GPU with CUDA support (for accelerated inference)
  - GPU drivers installed (if using CUDA)

#### Environment Setup
1. `.env` File (in project root)
   - Required variables:
     ```env
     MARIADB_ROOT_PASSWORD=BaFu5wRZGcO~S6*X^h0DM/TvE
     MARIADB_USER=chatbot
     MARIADB_PASSWORD=p/Le5,yh4VInU54vL=^1s521q!.V3t
     MARIADB_DATABASE=chatbot
     ```
   - Optional variables:
     ```env
     MAX_OUTPUT_LENGTH=   # Default: 6000
     TOP_K_CHUNKS=        # Default: 10
     LLM_N_CTX=           # Default: 4608
     LLM_N_BATCH=         # Default: 512
     LLM_MODEL_PATH=      # Default: /app/models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
     ```
   - Notes:
     - Adjust values based on your hardware (e.g., reduce `LLM_N_CTX` if OOM errors occur).
     - Ensure `LLM_MODEL_PATH` points to a valid `.gguf` file (or your preferred, llama.cpp supported model format).

2. Model File
   - Place your model in `/models/` (and update `LLM_MODEL_PATH` in `.env`).

### CPU-only deployment
- bash:
  ```bash
  DOCKER_BUILDKIT=1 docker compose up --build
  ```
- Windows PowerShell:
  ```bash
  $env:DOCKER_BUILDKIT=1; docker compose up --build
  ```
- Windows CMD:
  ```bash
  set DOCKER_BUILDKIT=1&& docker compose up --build
  ```
### GPU-accelerated deployment
- bash:
  ```bash
  DOCKER_BUILDKIT=1 docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build
  ```
- Windows PowerShell:
  ```bash
  $env:DOCKER_BUILDKIT=1; docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build
  ```
- Windows CMD:
  ```bash
  set DOCKER_BUILDKIT=1&& docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build
  ```
  
#### Troubleshooting
- Out of Memory (OOM):
  - Reduce `LLM_N_BATCH` (slower) or `LLM_N_CTX`, `MAX_OUTPUT_LENGTH`, `TOP_K_CHUNKS` (may worsen or even break the pipeline).
  - Use a smaller model (e.g., `TinyLlama` instead of `Ministral-3B`).
- CUDA Errors:
  - Verify `nvidia-smi` detects your GPU.
  - Ensure Docker has GPU access (`docker run --gpus all ...`).

## Access

- Frontend: http://localhost:8002
- Backend API: http://localhost:8001
- API documentation: http://localhost:8001/docs

## Supported domains

The system currently supports content extraction from four domains through the MWP microservice:

- <a href="it.wikipedia.org">it.wikipedia.org</a> -- general Italian knowledge
- www.ipsos.com -- market research and public opinion data
- www.raiplaysound.it -- Italian radio and podcast content
- www.marvel.com -- Marvel comics and MCU information

When a query cannot be assigned to a domain, a generic, domain-indefinite parse is requested to the MWP client as a means of fallback. This allows the LLM to answer extremely specific or niche questions for which no answer can be found on the provided domains.

## Key features

**Intelligent guardrail system**:
every user query passes through a validation layer that classifies it as allowed, ambiguous, or rejected. Ambiguous queries prompt the user for clarification before proceeding to search. The guardrail also determines which domain best matches the query intent, enabling automatic routing to the appropriate knowledge source.

**Multi-domain search with fallback**:
the system searches across a primary target domain and supplements results from Wikipedia when needed. If Startpage-based scraping returns no results, the Wikipedia OpenSearch API acts as a fallback retriever.

**Two-stage chunk selection**:
retrieved documents are chunked with overlap, embedded using a multilingual E5-small model (384 dimensions), and filtered by cosine similarity. The top candidates are then reranked by a cross-encoder (MS Marco MiniLM) for precise relevance scoring. This balances speed and accuracy within tight resource constraints.

**Transparent reliability scoring**:
every generated answer includes a mandatory reliability section (affidabilità). The LLM self-assesses how well the reference texts support its response, providing both a qualitative comment and an integer score from 1 to 5. The score can be used by users to gauge confidence in the generated content.

**Response streaming with progress feedback**:
server-sent events stream both status updates and answer tokens to the frontend. Users see real-time feedback during searching, data extraction, chunk selection, and answer generation phases, making latency from the local LLM more tolerable and the system behavior more transparent.

**Parse caching**:
parsed page content is stored in MariaDB keyed by URL. Subsequent queries referencing the same URLs skip the web parsing step entirely, reducing both latency and load on the MWP microservice.

**Conversation continuity**:
chat history is persisted across sessions. The system uses historical context to rewrite follow-up queries, resolve pronouns and implicit references, and maintain domain routing consistency within a conversation.

### Core components

**Backend** (FastAPI on port 8001)
- Query validation and intent analysis through structured LLM prompts
- Multi-domain URL retrieval via Startpage search and Wikipedia API
- Web content extraction through the integrated Minerva Web Parser microservice (MWP)
- Two-stage retrieval: bi-encoder cosine similarity followed by cross-encoder reranking
- Local LLM inference with streaming token generation via SSE
- Persistent chat history and URL parse caching through MariaDB

**Frontend** (FastAPI + Jinja2 on port 8002)
- Clean, responsive chat interface styled with Tailwind CSS
- Session management with conversation history browsing
- Real-time streaming of LLM responses with status indicators
- Toggle for forcing fresh web parsing versus cached content

**Database** (MariaDB 11)
- Session and message persistence with foreign key relationships
- Parsed URL content caching to avoid redundant web scraping
- Automatic session titling based on first user message

**LLM runtime** (llama.cpp)
- Serves LLM model with GPU acceleration support
- Model file retrieved from the _/models_ folder

## Architecture overview

The project follows a microservice-oriented architecture orchestrated through Docker compose, with a clear separation between the backend RAG pipeline and the frontend chat interface.

<pre>
User query 
    ↓ 
Guardrail validation (Ministral-3-3B-Instruct-2512-Q4_K_M *)
    ↓
Query rewriting (Ministral-3-3B-Instruct-2512-Q4_K_M *)
    ↓
URL retrieval (SSE scraper)
    ↓ 
Web parsing (MWP)
    ↓ 
Chunking (LangChain-RecursiveCharacterTextSplitter)
    ↓ 
Embedding  (intfloat/multilingual-e5-small)
    ↓ 
Reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
    ↓ 
Context assembly 
    ↓ 
LLM generation (Ministral-3-3B-Instruct-2512-Q4_K_M *)
    ↓ 
Streamed response
</pre>
\* tested with Ministral-3-3B-Instruct-2512-Q4_K_M, though no specific model constraint was applied

## Technology stack

- Backend: Python 3.12, FastAPI, Uvicorn
- Frontend: FastAPI, Jinja2, Tailwind CSS, 
- Database: MariaDB 11
- LLM inference: llama.cpp (tested with Ministral-3-3B-Instruct-2512-Q4_K_M from HuggingFace)
- Embeddings: intfloat/multilingual-e5-small via FastEmbed with ONNX runtime
- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 via SentenceTransformers
- Chunking: LangChain RecursiveCharacterTextSplitter
- Web parsing: **Minerva Web Parser** - <a href="https://github.com/fabio-pastore/minerva-web-parser">*MWP*</a> (custom-made parser to extract relevant information from given page URLs)
- Containerization: Docker, Docker-compose

### Directory structure


_domain-chatbot/_ <br>
_├── backend/_ <br>
_│ ├── src/_ <br>
_│ │ ├── api_server/_ # FastAPI application and SSE streaming <br>
_│ │ ├── db_manager/_ # MariaDB connection and query methods <br>
_│ │ ├── llm_manager/_ # LLM inference, prompt building, response handling <br>
_│ │ ├── query_handler/_ # Query processing pipeline orchestration <br>
_│ │ ├── rag/_ # Chunking, embedding, reranking, context generation <br>
_│ │ ├── url_retriever/_ # Search engine and Wikipedia URL retrieval <br>
_│ │ └── utils/_ # HTTP request utilities <br>
_│ ├── Dockerfile_ <br>
_│ └── requirements.txt_ <br> 
_├── frontend/_ <br>
_│ ├── src/_ <br>
_│ │ ├── frontend.py_ # FastAPI proxy and page routes_ <br>
_│ │ └── templates/_ <br>
_│ │ └── index.html_ # Chat interface <br>
_│ └── Dockerfile_ <br>
_├── docker-compose.yaml_ # Core services <br>
_├── docker-compose.gpu.yaml_ # GPU acceleration-specific docker compose <br>
_└── README.md_ <br>

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /chat | Send a query and stream the RAG response via SSE |
| GET | /sessions | List all chat sessions |
| POST | /sessions | Create a new chat session |
| GET | /sessions/{id}/messages | Retrieve message history for a session |
| DELETE | /sessions/{id} | Delete a session and its messages |
| DELETE | /clear_cache | Clear all parsed URL cache entries |
| GET | /health | Health check |

## Limitations and future work

~~The system is constrained if used with a 3B parameter model's reasoning capabilities, particularly for complex multi-hop questions.~~ Domain support is currently limited to the four configured websites. Fallback parsing may extract relevant information but may also introduce excessive noise, since the employed parser is generic. ~~Memory usage under Docker on Windows11 with 16 GB RAM (CPU-only setup) remains tight, and future iterations may migrate to llama.cpp for more efficient CPU inference and reduced container overhead.~~
