import os
from src.llm_manager.BaseLLMResponder import BaseLLMResponder
from llama_cpp import Llama # type: ignore
from src.llm_manager.BaseLLMResponder import BaseLLMResponder

class LlamaCppResponder(BaseLLMResponder):

    def __init__(self):
        """
        Initializes the LlamaCppResponder with the given env variables.
        """
        n_threads=int(os.getenv("LLM_N_THREADS", "0")) or None
        model_path=os.getenv("LLM_MODEL_PATH", "/app/models/Ministral-3B-Instruct-2512-Q4_K_M.gguf")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please download a Ministral 3B 4-bit GGUF model (or any other tbh) "
                "and place it at the configured path."
            )

        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_ctx=int(os.getenv("LLM_N_CTX", "4608")),
            n_threads=n_threads if n_threads is not None else (os.cpu_count() or 4),
            n_gpu_layers=int(os.getenv("LLM_N_GPU_LAYERS", "0")),
            n_batch=int(os.getenv("LLM_N_BATCH", "512")),
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