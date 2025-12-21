"""
LLM Providers package.

Provides unified access to different LLM backends.
"""

from src.providers.base import (
    BaseProvider,
    OllamaProvider,
    ProviderError,
    ProviderModelNotFoundError,
    ProviderRateLimitError,
    ProviderResponse,
    ProviderTimeoutError,
)
from src.providers.gemini import GeminiProvider
from src.providers.manager import ProviderManager, get_provider_manager

__all__ = [
    # Base
    "BaseProvider",
    "ProviderResponse",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderRateLimitError",
    "ProviderModelNotFoundError",
    # Implementations
    "OllamaProvider",
    "GeminiProvider",
    # Manager
    "ProviderManager",
    "get_provider_manager",
]
