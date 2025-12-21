"""
LLM Provider Integrations.

This module provides a unified interface for interacting with different LLM backends:
- OllamaProvider: Local LLM via Ollama (Granite 4.0 Nano)
- GeminiProvider: Google Gemini API (Flash + Pro)

The abstract BaseProvider class allows easy swapping and fallback behavior.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import httpx

from src.config import get_settings


@dataclass
class ProviderResponse:
    """
    Standard response from any LLM provider.
    
    Attributes:
        text: The generated text response
        model: The model that was used
        prompt_tokens: Number of tokens in the prompt (estimated)
        completion_tokens: Number of tokens in the response (estimated)
        latency_ms: Time taken for the request in milliseconds
        raw_response: The raw response from the provider (for debugging)
    """
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    raw_response: Optional[dict] = None


class ProviderError(Exception):
    """Base exception for provider errors."""
    def __init__(self, message: str, provider: str, retriable: bool = True):
        super().__init__(message)
        self.provider = provider
        self.retriable = retriable


class ProviderTimeoutError(ProviderError):
    """Request timed out."""
    pass


class ProviderRateLimitError(ProviderError):
    """Rate limit exceeded."""
    pass


class ProviderModelNotFoundError(ProviderError):
    """Model not available."""
    def __init__(self, message: str, provider: str):
        super().__init__(message, provider, retriable=False)


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement the `generate` method.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, model: str) -> ProviderResponse:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The user's prompt
            model: The specific model to use
            
        Returns:
            ProviderResponse with generated text and metadata
            
        Raises:
            ProviderError: If the request fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is available.
        
        Returns:
            True if the provider is healthy, False otherwise
        """
        pass
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token for English)."""
        return len(text) // 4


class OllamaProvider(BaseProvider):
    """
    Provider for local LLM via Ollama.
    
    Uses the Ollama REST API to generate completions.
    Default model: granite4:350m (Granite 4.0 Nano)
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 120):
        """
        Initialize OllamaProvider.
        
        Args:
            base_url: Ollama API URL (defaults to settings)
            timeout: Request timeout in seconds (default 120 for slow models)
        """
        settings = get_settings()
        self.base_url = base_url or settings.ollama_base_url
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    @property
    def name(self) -> str:
        return "ollama"
    
    async def generate(self, prompt: str, model: str) -> ProviderResponse:
        """
        Generate completion using Ollama API.
        
        Endpoint: POST /api/generate
        """
        start_time = time.perf_counter()
        
        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,  # Get complete response at once
                },
            )
            
            if response.status_code == 404:
                raise ProviderModelNotFoundError(
                    f"Model '{model}' not found in Ollama. Run: ollama run {model}",
                    self.name,
                )
            
            response.raise_for_status()
            data = response.json()
            
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            return ProviderResponse(
                text=data.get("response", ""),
                model=model,
                prompt_tokens=data.get("prompt_eval_count", self._estimate_tokens(prompt)),
                completion_tokens=data.get("eval_count", self._estimate_tokens(data.get("response", ""))),
                latency_ms=latency_ms,
                raw_response=data,
            )
            
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                f"Ollama request timed out after {self.timeout}s",
                self.name,
            )
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Ollama HTTP error: {e.response.status_code}",
                self.name,
            )
        except httpx.RequestError as e:
            raise ProviderError(
                f"Ollama connection error: {str(e)}",
                self.name,
            )
    
    async def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
