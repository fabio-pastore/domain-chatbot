import os
import threading
from src.llm_manager.LlamaCppResponder import LlamaCppResponder
from src.llm_manager.OpenAIResponder import OpenAIResponder
from openai import AuthenticationError, APIConnectionError, APITimeoutError, APIError # type: ignore
from src.llm_manager.BaseLLMResponder import BaseLLMResponder
from typing import Any


class LLMProviderManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._mode = None
        self._local_responder = None
        self._api_responder = None
        self._error = None
        self._model_info = ""

        provider = os.getenv("LLM_PROVIDER", "local").lower()
        if provider == "api":
            self._init_api()
        else:
            self._init_local()

    def _init_local(self) -> None:
        try:
            self._local_responder = LlamaCppResponder()
            self._mode = "local"
            self._error = None
            self._model_info = os.getenv("LLM_MODEL_PATH", "/models/Ministral-3B-Instruct-2512-Q4_K_M.gguf")
        except FileNotFoundError as e:
            self._local_responder = None
            self._mode = "local"
            self._error = str(e)
            self._model_info = "Model file not found"
        except Exception as e:
            self._local_responder = None
            self._mode = "local"
            self._error = str(e)
            self._model_info = "Failed to load local model"

    def _init_api(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "")
        model_name = os.getenv("OPENAI_MODEL_NAME", "")

        if not api_key or not base_url or not model_name:
            self._api_responder = None
            self._mode = "api"
            self._error = "Missing API configuration: OPENAI_API_KEY, OPENAI_BASE_URL, or OPENAI_MODEL_NAME"
            self._model_info = "Not configured"
            return

        try:
            self._api_responder = OpenAIResponder(api_key, base_url, model_name)
            self._mode = "api"
            self._error = None
            self._model_info = f"{model_name} @ {base_url}"
        except Exception as e:
            self._api_responder = None
            self._mode = "api"
            self._error = str(e)
            self._model_info = "Failed to initialize API client"

    def get_responder(self) -> BaseLLMResponder | None:
        if self._mode == "local":
            return self._local_responder
        elif self._mode == "api":
            return self._api_responder
        return None

    def get_status(self) -> dict[str, str|None]:
        return {
            "provider": self._mode,
            "error": self._error,
            "model_info": self._model_info,
        }

    def switch_to_local(self) -> dict[str, Any]:
        if self._local_responder is None:
            try:
                self._local_responder = LlamaCppResponder()
            except FileNotFoundError as e:
                self._error = str(e)
                self._model_info = "Model file not found"
                return {"success": False, "error": str(e)}
            except Exception as e:
                self._error = str(e)
                self._model_info = "Failed to load local model"
                return {"success": False, "error": str(e)}

        self._mode = "local"
        self._error = None
        self._model_info = os.getenv("LLM_MODEL_PATH", "/models/Ministral-3B-Instruct-2512-Q4_K_M.gguf")
        return {"success": True}

    def switch_to_api(self, api_key: str, base_url: str, model_name: str) -> dict[str, Any]:
        if not api_key or not base_url or not model_name:
            return {"success": False, "error": "Missing required fields: api_key, base_url, model_name"}
        try:
            self._api_responder = OpenAIResponder(api_key, base_url, model_name)
            self._mode = "api"
            self._error = None
            self._model_info = f"{model_name} @ {base_url}"
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_api_connection(self, api_key: str, base_url: str, model_name: str) -> dict[str, str|bool|None]:
        if not api_key or not base_url or not model_name:
            return {"success": False, "error": "Missing required fields: api_key, base_url, model_name"}
        try:
            test_client = OpenAIResponder(api_key, base_url, model_name, timeout=15.0)
            response = test_client.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "OK"}],
                max_tokens=1,
            )
            if not response.choices:
                return {"success": False, "error": "Model returned an empty response."}
            return {"success": True, "error": None}
        except AuthenticationError:
            return {"success": False, "error": "Authentication failed: invalid API key."}
        except APITimeoutError:
            return {"success": False, "error": "Connection timed out after 15 seconds."}
        except APIConnectionError:
            return {"success": False, "error": f"Connection failed: unable to reach {base_url}."}
        except APIError as e:
            return {"success": False, "error": f"API error: {e.message if hasattr(e, 'message') else str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


llm_provider_manager = LLMProviderManager()
