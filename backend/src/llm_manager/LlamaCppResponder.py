import os
from abc import ABC
from llama_cpp import Llama

class LlamaCppResponder(ABC):

    def __init__(self, model_path: str, n_ctx: int = 8192, n_threads: int = None, n_gpu_layers: int = 0, n_batch: int = 512):
        """
        Initializes the LlamaCppResponder with the given GGUF model path.

        Args:
            model_path (str): Path to the GGUF model file.
            n_ctx (int): Context window size. Defaults to 8192.
            n_threads (int): Number of CPU threads to use. Defaults to os.cpu_count().
            n_gpu_layers (int): Number of layers to offload to GPU (-1 for all). Defaults to 0 (CPU only).
            n_batch (int): Prompt processing batch size. Defaults to 512.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please download a Ministral 3B 4-bit GGUF model (e.g., from HuggingFace) "
                "and place it at the configured path."
            )

        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads if n_threads is not None else (os.cpu_count() or 4),
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=False
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Calls the local llama.cpp model with the given prompt and returns the response.

        Args:
            prompt (str): The input prompt to send to the model.

        Returns:
            str: The response from the model, or an empty string in case of an error.
        """
        try:
            output = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=-1,
                temperature=0.7,
            )
            return output["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error calling llama.cpp model: {e}")
            return ""

    def _stream_llm(self, prompt: str):
        """
        Calls the local llama.cpp model with the given prompt and yields response tokens.

        Args:
            prompt (str): The input prompt to send to the model.

        Yields:
            str: Tokens from the model.
        """
        try:
            stream = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=-1,
                temperature=0.7,
                stream=True
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
        except Exception as e:
            print(f"Error streaming from llama.cpp model: {e}")
            yield ""
