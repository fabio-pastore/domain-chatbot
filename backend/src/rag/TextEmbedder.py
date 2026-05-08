from src.ollama_manager.EmbeddingResponder import embedding_responder

class TextEmbedder:

    @staticmethod
    def embed_text(text: str) -> list[float]:
        response = embedding_responder._call_ollama(text, embed=True)
        return response