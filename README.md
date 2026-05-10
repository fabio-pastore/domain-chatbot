## Sapienza-DC

This project is focused on the use of a local LLM model to answer user queries RAG, done by extracting relevant data and parsing URLs which contain it.
The project is still in an early development stage.

To run on CPU only, build with the following command
```DOCKER_BUILDKIT=1 docker compose up --build``` (Bash)
```$env:DOCKER_BUILDKIT=1; docker compose up --build``` (Windows PowerShell)
```set DOCKER_BUILDKIT=1 && docker compose up --build``` (Windows CMD)
If you wish to use your GPU for increased performance, build with
```DOCKER_BUILDKIT=1 docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build``` (Bash)
```$env:DOCKER_BUILDKIT=1; docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build``` (Windows PowerShell)
```set DOCKER_BUILDKIT=1 && docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build``` (Windows CMD)