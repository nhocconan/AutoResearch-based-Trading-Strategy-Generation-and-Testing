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
from pathlib import Path
from typing import Optional

import yaml


def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(self, provider: Optional[str] = None, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.llm_config = config["llm"]
        self.provider = provider or self.llm_config["default_provider"]
        self.provider_config = self.llm_config["providers"][self.provider]

        self._client = None
        self._init_client()

    def _get_api_key(self) -> str:
        env_var = self.provider_config["api_key_env"]
        key = os.environ.get(env_var)
        if not key:
            raise ValueError(
                f"API key not found. Set {env_var} environment variable."
            )
        return key

    def _get_base_url(self) -> Optional[str]:
        # Config override takes priority
        if self.provider_config.get("base_url"):
            return self.provider_config["base_url"]
        # Then environment variable
        env_var = self.provider_config.get("base_url_env")
        if env_var:
            return os.environ.get(env_var)
        return None

    def _init_client(self):
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "gemini":
            self._init_gemini()
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
        self._client = genai.GenerativeModel(self.provider_config["model"])

    def chat(
        self,
        message: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a message and get a response."""
        temp = temperature or self.provider_config.get("temperature", 0.7)
        tokens = max_tokens or self.provider_config.get("max_tokens", 4096)

        if self.provider == "openai":
            return self._chat_openai(message, system, temp, tokens)
        elif self.provider == "anthropic":
            return self._chat_anthropic(message, system, temp, tokens)
        elif self.provider == "gemini":
            return self._chat_gemini(message, system, temp, tokens)

    def _chat_openai(self, message: str, system: str, temp: float, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        response = self._client.chat.completions.create(
            model=self.provider_config["model"],
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _chat_anthropic(self, message: str, system: str, temp: float, max_tokens: int) -> str:
        kwargs = {
            "model": self.provider_config["model"],
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
