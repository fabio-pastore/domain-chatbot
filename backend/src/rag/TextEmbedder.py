from src.ollama_manager.EmbeddingResponder import embedding_responder

class TextEmbedder:

    @staticmethod
    def embed_batch(texts_to_embed: list[str]) -> list[list[float]]:
        response = embedding_responder._call_ollama(texts_to_embed, embed=True)
        return response
    