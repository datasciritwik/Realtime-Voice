import os
import sys
import json
import time
import uuid
import logging
import subprocess
from threading import Lock
from typing import Optional, Generator, Dict, Any, List, Union

# --- Library Dependencies ---
try:
    import requests
    from requests import Session
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("🤖⚠️ requests library not installed. Ollama backend (direct HTTP) will not function.")
    if sys.version_info >= (3, 9): Session = Any | None
    else: Session = Optional[Any]

try:
    from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    class APIError(Exception): pass
    class APITimeoutError(APIError): pass
    class RateLimitError(APIError): pass
    class APIConnectionError(APIError): pass
    logging.warning("🤖⚠️ openai library not installed. OpenAI/LMStudio backends will not function.")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None
    logging.warning("🤖⚠️ groq library not installed. Groq backend will not function.")

# --- Configuration ---
# Default configuration from environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure logger
logger = logging.getLogger(__name__)
# Basic configuration if not already configured by parent
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def _check_ollama_connection() -> bool:
    """
    Attempts to run the 'ollama ps' command via subprocess.
    """
    try:
        logger.info("🤖🩺 Attempting to run 'ollama ps' to check server status...")
        result = subprocess.run(["ollama", "ps"], check=True, capture_output=True, text=True, timeout=10.0)
        logger.info(f"🤖🩺 'ollama ps' executed successfully. Output:\n{result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.error("🤖💥 'ollama ps' command not found. Make sure Ollama is installed and in your PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"🤖💥 'ollama ps' command failed with exit code {e.returncode}:")
        if e.stderr:
            logger.error(f"   stderr: {e.stderr.strip()}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("🤖💥 'ollama ps' command timed out after 10 seconds.")
        return False
    except Exception as e:
        logger.error(f"🤖💥 An unexpected error occurred while running 'ollama ps': {e}")
        return False

class LLM:
    """
    Provides a unified interface for interacting with various LLM backends.
    """
    SUPPORTED_BACKENDS = ["ollama", "openai", "lmstudio", "groq"]

    def __init__(
        self,
        backend: str,
        model: str,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        no_think: bool = False,
    ):
        logger.info(f"🤖⚙️ Initializing LLM with backend: {backend}, model: {model}, system_prompt: {system_prompt}")
        self.backend = backend.lower()
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Supported: {self.SUPPORTED_BACKENDS}")

        if self.backend == "ollama" and not REQUESTS_AVAILABLE:
             raise ImportError("requests library is required for the 'ollama' backend but not installed.")
        if self.backend in ["openai", "lmstudio"] and not OPENAI_AVAILABLE:
             raise ImportError("openai library is required for the 'openai'/'lmstudio' backends but not installed.")
        if self.backend == "groq" and not GROQ_AVAILABLE:
             raise ImportError("groq library is required for the 'groq' backend but not installed.")

        self.model = model
        self.system_prompt = system_prompt
        self._api_key = api_key
        self._base_url = base_url
        self.no_think = no_think

        self.client: Optional[Any] = None
        self.ollama_session: Optional[Session] = None
        self._client_initialized: bool = False
        self._client_init_lock = Lock()
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._requests_lock = Lock()
        self._ollama_connection_ok: bool = False

        self.effective_openai_key = self._api_key or OPENAI_API_KEY
        self.effective_ollama_url = self._base_url or OLLAMA_BASE_URL if self.backend == "ollama" else None
        self.effective_lmstudio_url = self._base_url or LMSTUDIO_BASE_URL if self.backend == "lmstudio" else None
        self.effective_openai_base_url = self._base_url if self.backend == "openai" and self._base_url else None

    def _create_openai_client(self, api_key: Optional[str], base_url: Optional[str] = None) -> Any:
        if not api_key and self.backend == "openai":
             logger.warning("🤖⚠️ No API key provided for OpenAI backend. Calls may fail.")
        return OpenAI(api_key=api_key, base_url=base_url)

    def _create_groq_client(self, api_key: Optional[str]) -> Any:
        if not api_key:
             api_key = GROQ_API_KEY
        if not api_key:
             raise ValueError("GROQ_API_KEY not found in environment variables or passed to constructor.")
        return Groq(api_key=api_key)

    def _lazy_initialize_clients(self) -> bool:
        if self._client_initialized:
            return True

        with self._client_init_lock:
            if self._client_initialized:
                return True

            try:
                if self.backend == "ollama":
                    if not self.effective_ollama_url:
                        raise ValueError("Ollama base URL not configured.")
                    
                    # Check connection
                    try:
                        test_url = f"{self.effective_ollama_url}/api/tags"
                        requests.get(test_url, timeout=2.0)
                        self._ollama_connection_ok = True
                    except Exception:
                        logger.warning(f"🤖⚠️ Could not connect to Ollama at {self.effective_ollama_url}. Trying 'ollama ps'...")
                        if _check_ollama_connection():
                             self._ollama_connection_ok = True
                        else:
                             logger.error("🤖💥 Ollama connection failed and 'ollama ps' check failed.")
                             return False

                    self.ollama_session = requests.Session()
                    self._client_initialized = True
                    return True

                elif self.backend == "openai":
                    self.client = self._create_openai_client(self.effective_openai_key, self.effective_openai_base_url)
                    self._client_initialized = True
                    return True

                elif self.backend == "lmstudio":
                    self.client = self._create_openai_client("lm-studio", self.effective_lmstudio_url)
                    self._client_initialized = True
                    return True
                
                elif self.backend == "groq":
                    self.client = self._create_groq_client(self._api_key)
                    self._client_initialized = True
                    return True

            except Exception as e:
                logger.error(f"🤖💥 Failed to initialize client for {self.backend}: {e}")
                return False
            
        return False

    def prewarm(self, max_retries: int = 0) -> bool:
        return self._lazy_initialize_clients()

    def _register_request(self, request_id: str, backend_type: str, stream_object: Any):
        with self._requests_lock:
            self._active_requests[request_id] = {
                "type": backend_type,
                "stream": stream_object,
                "start_time": time.time()
            }

    def _cancel_single_request_unsafe(self, request_id: str):
        if request_id in self._active_requests:
            req_info = self._active_requests[request_id]
            backend_type = req_info["type"]
            stream = req_info["stream"]
            
            try:
                if backend_type == "ollama":
                    if isinstance(stream, requests.Response):
                        stream.close()
                elif backend_type in ["openai", "lmstudio", "groq"]:
                    if hasattr(stream, "close"):
                        stream.close()
            except Exception as e:
                logger.warning(f"Error closing stream for {request_id}: {e}")
            
            del self._active_requests[request_id]

    def cancel_request(self, request_id: str):
        with self._requests_lock:
            self._cancel_single_request_unsafe(request_id)

    def _yield_openai_chunks(self, stream: Any, request_id: str) -> Generator[str, None, None]:
        token_count = 0
        try:
            for chunk in stream:
                with self._requests_lock:
                    if request_id not in self._active_requests:
                        break
                
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = delta.content
                    if content:
                        token_count += 1
                        yield content
        except Exception as e:
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = request_id not in self._active_requests
            if not is_cancelled:
                logger.error(f"Error in OpenAI stream: {e}")
                raise
        finally:
            if hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception:
                    pass

    def _yield_ollama_chunks(self, response: requests.Response, request_id: str) -> Generator[str, None, None]:
        token_count = 0
        buffer = ""
        processed_done = False
        try:
            for chunk_bytes in response.iter_content(chunk_size=None):
                with self._requests_lock:
                    if request_id not in self._active_requests:
                        break
                
                if not chunk_bytes:
                    continue
                
                buffer += chunk_bytes.decode('utf-8')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip():
                        continue
                    
                    try:
                        chunk = json.loads(line)
                        if chunk.get('error'):
                            raise RuntimeError(f"Ollama error: {chunk['error']}")
                        
                        content = chunk.get('message', {}).get('content')
                        if content:
                            token_count += 1
                            yield content
                        
                        if chunk.get('done'):
                            processed_done = True
                            break
                    except json.JSONDecodeError:
                        continue
                
                if processed_done:
                    break
                    
        except Exception as e:
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = request_id not in self._active_requests
            if not is_cancelled:
                logger.error(f"Error in Ollama stream: {e}")
                raise
        finally:
            response.close()

    def generate(self, text: str, history: List[Dict[str, str]] = [], use_system_prompt: bool = True, request_id: Optional[str] = None, **kwargs: Any) -> Generator[str, None, None]:
        req_id = request_id or str(uuid.uuid4())
        
        if not self._lazy_initialize_clients():
            raise ConnectionError(f"Could not initialize {self.backend} client.")

        messages = []
        if use_system_prompt and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.extend(history)
        if text:
            messages.append({"role": "user", "content": text})

        try:
            if self.backend in ["openai", "lmstudio", "groq"]:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    **kwargs
                )
                self._register_request(req_id, self.backend, stream)
                yield from self._yield_openai_chunks(stream, req_id)
            
            elif self.backend == "ollama":
                url = f"{self.effective_ollama_url}/api/chat"
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": kwargs
                }
                response = self.ollama_session.post(url, json=payload, stream=True)
                response.raise_for_status()
                self._register_request(req_id, "ollama", response)
                yield from self._yield_ollama_chunks(response, req_id)
            
            else:
                 raise ValueError(f"Backend '{self.backend}' generation logic not implemented.")
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
        finally:
            self.cancel_request(req_id)

    def measure_inference_time(self, num_tokens: int = 10, **kwargs: Any) -> Optional[float]:
        if num_tokens <= 0: return None
        if not self._lazy_initialize_clients(): return None
        
        req_id = f"measure-{uuid.uuid4()}"
        history = [{"role": "user", "content": "Count to 20"}]
        
        try:
            start_time = time.time()
            generator = self.generate(text="", history=history, use_system_prompt=False, request_id=req_id, **kwargs)
            
            count = 0
            for _ in generator:
                count += 1
                if count >= num_tokens:
                    break
            
            end_time = time.time()
            return (end_time - start_time) * 1000
            
        except Exception:
            return None

class LLMGenerationContext:
    def __init__(self, llm_instance: LLM, text: str, **kwargs):
        self.llm = llm_instance
        self.text = text
        self.kwargs = kwargs
        self.request_id = str(uuid.uuid4())
        self.generator = None

    def __enter__(self):
        self.generator = self.llm.generate(self.text, request_id=self.request_id, **self.kwargs)
        return self.generator

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.llm.cancel_request(self.request_id)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("LLM Module Test")