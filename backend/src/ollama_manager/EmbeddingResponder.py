import os
from src.ollama_manager.OllamaResponder import OllamaResponder

class EmbeddingResponder(OllamaResponder):
    def __init__(self):
        super().__init__(
            ollama_url = os.getenv("OLLAMA_EMBEDDING_URL", "http://host.docker.internal:11434/api/embed"),
            ollama_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        )

embedding_responder = EmbeddingResponder()

    