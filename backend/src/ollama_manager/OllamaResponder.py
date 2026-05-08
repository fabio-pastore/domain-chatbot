from abc import ABC
import requests

class OllamaResponder(ABC):

    def __init__(self, ollama_url: str, ollama_model: str):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def _call_ollama(self, prompt: str, embed: bool = False) -> str:
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
                return data.get("embedding", "") # list[float]
            else:
                return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return ""