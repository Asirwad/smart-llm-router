"""
Provider Manager.

Manages multiple LLM providers and handles:
- Provider selection based on model tier
- Retry logic with exponential backoff
- Fallback chain (local → external)
"""

import asyncio
from typing import Optional

from src.config import get_settings
from src.core.router import DifficultyTier
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


class ProviderManager:
    """
    Manages LLM providers and handles routing, retries, and fallback.
    
    Provider Mapping:
    - SIMPLE → Ollama (Granite 4.0 Nano)
    - MEDIUM → Gemini Flash
    - COMPLEX → Gemini Pro
    
    Fallback Chain:
    - If local model fails → escalate to Gemini Flash
    - If Gemini Flash fails → escalate to Gemini Pro
    """
    
    def __init__(self):
        """Initialize providers."""
        self.settings = get_settings()
        
        # Initialize providers lazily
        self._ollama: Optional[OllamaProvider] = None
        self._gemini: Optional[GeminiProvider] = None
        
        # Tier to model mapping
        self._tier_models = {
            DifficultyTier.SIMPLE: self.settings.ollama_model,
            DifficultyTier.MEDIUM: self.settings.gemini_flash_model,
            DifficultyTier.COMPLEX: self.settings.gemini_pro_model,
        }
        
        # Tier to provider mapping
        self._tier_providers = {
            DifficultyTier.SIMPLE: "ollama",
            DifficultyTier.MEDIUM: "gemini",
            DifficultyTier.COMPLEX: "gemini",
        }
        
        # Fallback chain: tier → next tier to try
        self._fallback_chain = {
            DifficultyTier.SIMPLE: DifficultyTier.MEDIUM,
            DifficultyTier.MEDIUM: DifficultyTier.COMPLEX,
            DifficultyTier.COMPLEX: None,  # No fallback from complex
        }
    
    @property
    def ollama(self) -> OllamaProvider:
        """Get or create Ollama provider."""
        if self._ollama is None:
            self._ollama = OllamaProvider()
        return self._ollama
    
    @property
    def gemini(self) -> GeminiProvider:
        """Get or create Gemini provider."""
        if self._gemini is None:
            self._gemini = GeminiProvider()
        return self._gemini
    
    def get_provider(self, tier: DifficultyTier) -> BaseProvider:
        """Get the appropriate provider for a tier."""
        provider_name = self._tier_providers[tier]
        if provider_name == "ollama":
            return self.ollama
        else:
            return self.gemini
    
    def get_model(self, tier: DifficultyTier) -> str:
        """Get the model name for a tier."""
        return self._tier_models[tier]
    
    async def generate(
        self,
        prompt: str,
        tier: DifficultyTier,
        max_retries: int = 2,
        allow_fallback: bool = True,
    ) -> tuple[ProviderResponse, DifficultyTier]:
        """
        Generate completion with retry and fallback logic.
        
        Args:
            prompt: The user's prompt
            tier: The initial tier to try
            max_retries: Number of retries before fallback
            allow_fallback: Whether to escalate on failure
            
        Returns:
            Tuple of (ProviderResponse, actual_tier_used)
            
        Raises:
            ProviderError: If all providers fail
        """
        current_tier = tier
        last_error: Optional[Exception] = None
        
        while current_tier is not None:
            provider = self.get_provider(current_tier)
            model = self.get_model(current_tier)
            
            # Try with retries
            for attempt in range(max_retries + 1):
                try:
                    response = await provider.generate(prompt, model)
                    return response, current_tier
                    
                except ProviderModelNotFoundError:
                    # No point retrying if model doesn't exist
                    last_error = ProviderError(
                        f"Model {model} not found. Run: ollama run {model}",
                        provider.name,
                        retriable=False,
                    )
                    break
                    
                except ProviderRateLimitError as e:
                    # Wait longer for rate limits
                    last_error = e
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** (attempt + 2))  # 4, 8, 16 seconds
                    continue
                    
                except ProviderTimeoutError as e:
                    # Timeouts might succeed on retry
                    last_error = e
                    if attempt < max_retries:
                        await asyncio.sleep(1)
                    continue
                    
                except ProviderError as e:
                    last_error = e
                    if e.retriable and attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)  # 1, 2, 4 seconds
                        continue
                    break
            
            # Failed after retries, try fallback
            if allow_fallback:
                next_tier = self._fallback_chain.get(current_tier)
                if next_tier:
                    print(f"⚠️ Fallback: {current_tier.value} → {next_tier.value} (error: {last_error})")
                    current_tier = next_tier
                    continue
            
            # No more fallbacks
            break
        
        # All attempts failed
        raise last_error or ProviderError("All providers failed", "unknown")
    
    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers."""
        return {
            "ollama": await self.ollama.health_check(),
            "gemini": await self.gemini.health_check(),
        }
    
    async def close(self):
        """Close all provider connections."""
        if self._ollama:
            await self._ollama.close()
        if self._gemini:
            await self._gemini.close()


# Singleton instance
_manager: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    """Get the singleton ProviderManager instance."""
    global _manager
    if _manager is None:
        _manager = ProviderManager()
    return _manager
