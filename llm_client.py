#!/usr/bin/env python3
"""
llm_client.py - Multi-Provider LLM Client
==========================================
Unified interface for OpenAI-compatible, Anthropic, and Google Gemini APIs.
Supports custom endpoints and models via config.yaml or environment variables.

Usage:
    from llm_client import LLMClient
    client = LLMClient()  # Uses default provider from config
    response = client.chat("Analyze this trading strategy...")

    # Or specify provider
    client = LLMClient(provider="gemini")
"""

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml

# Default timeout for LLM calls (seconds). Prevents infinite hangs.
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "180"))

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm_call")


def _load_local_dotenv() -> None:
    """Load repo-local .env without overriding already-exported environment variables."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


class LLMTimeoutError(TimeoutError):
    """Raised when an LLM call exceeds LLM_TIMEOUT seconds."""
    pass


def _run_with_timeout(fn, timeout: int):
    """Run fn() in a thread; raise LLMTimeoutError if it exceeds timeout seconds."""
    future = _EXECUTOR.submit(fn)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        raise LLMTimeoutError(f"LLM call timed out after {timeout}s")


def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _is_local_ollama_url(url: Optional[str]) -> bool:
    if not url:
        return True
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in {"", "localhost", "127.0.0.1", "::1"}


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(
        self,
        provider: Optional[str] = None,
        config: Optional[dict] = None,
        model_override: Optional[str] = None,
    ):
        _load_local_dotenv()
        if config is None:
            config = load_config()

        self.llm_config = config["llm"]
        self.provider = provider or self.llm_config["default_provider"]
        self.provider_config = self.llm_config["providers"][self.provider]
        self.model_override = model_override

        self._client = None
        self._init_client()

    def _get_api_key(self) -> str:
        env_var = self.provider_config["api_key_env"]
        key = os.environ.get(env_var)
        if self.provider == "ollama" and not key and _is_local_ollama_url(self._get_base_url()):
            return ""
        if not key:
            raise ValueError(
                f"API key not found. Set {env_var} environment variable."
            )
        return key

    def _get_base_url(self) -> Optional[str]:
        # Environment override should win so deployments can switch local/cloud
        # without editing config.yaml.
        env_var = self.provider_config.get("base_url_env")
        if env_var:
            env_val = os.environ.get(env_var)
            if env_val:
                return env_val
        # Fall back to config
        if self.provider_config.get("base_url"):
            return self.provider_config["base_url"]
        return None

    def _get_model(self) -> str:
        if self.model_override:
            return self.model_override
        # Env var override (e.g. OPENAI_MODEL=qwen3-235b-a22b)
        model_env = self.provider_config.get("model_env")
        if model_env:
            env_val = os.environ.get(model_env)
            if env_val:
                return env_val
        return self.provider_config["model"]

    def _get_timeout(self) -> int:
        return int(self.provider_config.get("timeout", LLM_TIMEOUT))

    def _get_fallback_model(self) -> Optional[str]:
        """Get fallback model for rate-limit scenarios (openai provider only)."""
        fallback_env = self.provider_config.get("fallback_model_env")
        if fallback_env:
            return os.environ.get(fallback_env)
        return None

    def _init_client(self):
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _init_openai(self):
        import openai
        kwargs = {"api_key": self._get_api_key()}
        base_url = self._get_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)

    def _init_anthropic(self):
        import anthropic
        kwargs = {"api_key": self._get_api_key()}
        base_url = self._get_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)

    def _init_gemini(self):
        import google.generativeai as genai
        genai.configure(api_key=self._get_api_key())
        self._client = genai.GenerativeModel(self._get_model())

    def _init_ollama(self):
        import requests
        self._client = requests.Session()
        headers = {"Content-Type": "application/json"}
        api_key = self._get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client.headers.update(headers)

    def chat(
        self,
        message: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Send a message and get a response. Raises LLMTimeoutError if timeout exceeded."""
        temp = temperature or self.provider_config.get("temperature", 0.7)
        tokens = max_tokens or self.provider_config.get("max_tokens", 4096)
        t = timeout or self._get_timeout()

        if self.provider == "openai":
            return _run_with_timeout(lambda: self._chat_openai(message, system, temp, tokens), t)
        elif self.provider == "anthropic":
            return _run_with_timeout(lambda: self._chat_anthropic(message, system, temp, tokens), t)
        elif self.provider == "gemini":
            return _run_with_timeout(lambda: self._chat_gemini(message, system, temp, tokens), t)
        elif self.provider == "ollama":
            return _run_with_timeout(lambda: self._chat_ollama(message, system, temp, tokens), t)

    def _chat_openai(self, message: str, system: str, temp: float, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        response = self._client.chat.completions.create(
            model=self._get_model(),
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _chat_anthropic(self, message: str, system: str, temp: float, max_tokens: int) -> str:
        kwargs = {
            "model": self._get_model(),
            "max_tokens": max_tokens,
            "temperature": temp,
            "messages": [{"role": "user", "content": message}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def _chat_gemini(self, message: str, system: str, temp: float, max_tokens: int) -> str:
        prompt = f"{system}\n\n{message}" if system else message

        response = self._client.generate_content(
            prompt,
            generation_config={
                "temperature": temp,
                "max_output_tokens": max_tokens,
            },
        )
        return response.text

    def _chat_ollama(self, message: str, system: str, temp: float, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        base_url = self._get_base_url() or "http://127.0.0.1:11434/api/chat"
        primary_model = self._get_model()
        fallbacks = self.provider_config.get("fallback_models", [])
        models_to_try = [primary_model] + fallbacks
        extra_options = self.provider_config.get("options", {})

        last_err = None
        for model in models_to_try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temp,
                    "num_predict": max_tokens,
                    **extra_options,
                },
            }
            try:
                resp = self._client.post(base_url, json=payload, timeout=self._get_timeout())
                resp.raise_for_status()
                data = resp.json()
                if model != primary_model:
                    print(f"  [OLLAMA] {primary_model} unavailable, using {model}")
                return data["message"]["content"]
            except Exception as e:
                last_err = e
                status = getattr(getattr(e, 'response', None), 'status_code', None)
                if status in (503, 502, 500, 429):
                    print(f"  [OLLAMA] {model} returned {status}, trying next fallback...")
                    continue
                raise  # non-retriable error (auth, 400, etc.)

        raise last_err  # all models failed


def test_connection(provider: Optional[str] = None):
    """Test LLM connection with a simple query."""
    try:
        client = LLMClient(provider=provider)
        response = client.chat("Say 'hello' in one word.")
        print(f"[{client.provider}] Connected. Response: {response.strip()}")
        return True
    except Exception as e:
        print(f"[{provider or 'default'}] Connection failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    provider = sys.argv[1] if len(sys.argv) > 1 else None
    test_connection(provider)
