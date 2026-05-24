"""Kimi API client module.

Provides a thin wrapper around the MoonshotAI Kimi HTTP API,
handling authentication, request construction, and streaming responses.
"""

from __future__ import annotations

import os
import json
from typing import Generator, Optional

import httpx

API_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "moonshot-v1-8k"
DEFAULT_TIMEOUT = 120.0


class KimiClientError(Exception):
    """Raised when the Kimi API returns an error response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}")


class KimiClient:
    """Synchronous Kimi API client.

    Args:
        api_key: Moonshot API key.  Falls back to the ``MOONSHOT_API_KEY``
            environment variable when not supplied explicitly.
        model: Model identifier to use for chat completions.
        base_url: Override the default API base URL (useful for proxies /
            local mirrors).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = API_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        resolved_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key provided.  Set MOONSHOT_API_KEY or pass api_key=..."
            )
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        # Raised default temperature slightly from 0.3 to 0.7 — I find
        # the responses feel more natural for my personal use cases.
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str | Generator[str, None, None]:
        """Send a chat completion request.

        Args:
            messages: Conversation history in OpenAI message format.
            stream: When *True* returns a generator that yields text chunks
                as they arrive; otherwise blocks and returns the full reply.
            temperature: Sampling temperature (0 – 1).
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant reply as a plain string, or a generator of string
            chunks when ``stream=True``.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if stream:
           
