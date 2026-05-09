from abc import ABC
import requests

class OllamaResponder(ABC):

    def __init__(self, ollama_url: str, ollama_model: str):
        """
        Initializes the OllamaResponder with the given Ollama URL and model.

        Args:
            ollama_url (str): The URL of the Ollama service.
            ollama_model (str): The model to use for generating responses.
        """
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def _call_ollama(self, prompt: str, embed: bool = False) -> str | list[float]:
        """
        Calls the Ollama service with the given prompt and returns the response.

        Args:
            prompt (str): The input prompt to send to the Ollama service.
            embed (bool, optional): Whether to return an embedding instead of a response. Defaults to False.

        Returns:
            str | list[float]: The response from the Ollama service, or an empty string in case of an error.
        """
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if embed:
                return data.get("embedding", []) # list[float]
            else:
                return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return ""