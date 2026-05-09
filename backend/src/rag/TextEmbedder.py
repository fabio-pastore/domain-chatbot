from src.ollama_manager.EmbeddingResponder import embedding_responder

class TextEmbedder:

    @staticmethod
    def embed_text(text: str) -> list[float]:
        """Embeds the given text into a vector of floats using Ollama.

        Args:
            text (str): The text to be embedded.
        Returns:
            list[float]: A list of floats representing the text embedding.
        """
        response = embedding_responder._call_ollama(text, embed=True)
        return response
