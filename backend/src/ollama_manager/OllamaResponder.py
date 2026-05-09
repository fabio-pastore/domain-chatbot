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
        } if (not embed) else {
            "model": self.ollama_model,
            "input": prompt, # necessary for newer "embed" endpoint
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            if embed:
                return data.get("embeddings", []) # list[list[float]]
            else:
                return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return ""

    def _stream_ollama(self, prompt: str):
        """
        Calls the Ollama service with the given prompt and yields the response tokens.

        Args:
            prompt (str): The input prompt to send to the Ollama service.

        Yields:
            str: Tokens from the Ollama service.
        """
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": True
        }
        
        try:
            with requests.post(self.ollama_url, json=payload, stream=True, timeout=120) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        yield data.get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            yield ""