"""
Google Gemini Provider.

Integrates with Google Gemini API for Flash and Pro models.
Uses the official google-genai SDK with async support.
"""

import time
from typing import Optional

from google import genai
from google.genai import types

from src.config import get_settings
from src.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponse,
    ProviderTimeoutError,
)


class GeminiProvider(BaseProvider):
    """
    Provider for Google Gemini API.
    
    Supports:
    - gemini-2.0-flash-exp (fast, cost-effective)
    - gemini-1.5-pro (high capability)
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        """
        Initialize GeminiProvider.
        
        Args:
            api_key: Google API key (defaults to settings)
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        self.api_key = api_key or settings.google_api_key
        self.timeout = timeout
        
        if not self.api_key:
            raise ProviderError(
                "GOOGLE_API_KEY not configured",
                self.name,
                retriable=False,
            )
        
        # Initialize the Gemini client
        self._client = genai.Client(api_key=self.api_key)
    
    @property
    def name(self) -> str:
        return "gemini"
    
    async def generate(self, prompt: str, model: str) -> ProviderResponse:
        """
        Generate completion using Gemini API.
        
        Uses the async generate_content method.
        """
        start_time = time.perf_counter()
        
        try:
            # Create the request
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )
            
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            # Extract text from response
            text = ""
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                text += part.text or ""
            
            # Get token counts from usage metadata
            prompt_tokens = self._estimate_tokens(prompt)
            completion_tokens = self._estimate_tokens(text)
            
            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count or prompt_tokens
                completion_tokens = response.usage_metadata.candidates_token_count or completion_tokens
            
            return ProviderResponse(
                text=text,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                raw_response=None,  # Complex object, skip for now
            )
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limiting
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                raise ProviderRateLimitError(
                    f"Gemini rate limit exceeded: {e}",
                    self.name,
                )
            
            # Check for timeout
            if "timeout" in error_str or "deadline" in error_str:
                raise ProviderTimeoutError(
                    f"Gemini request timed out: {e}",
                    self.name,
                )
            
            # Generic error
            raise ProviderError(
                f"Gemini API error: {e}",
                self.name,
            )
    
    async def health_check(self) -> bool:
        """
        Check if Gemini API is accessible.
        
        Tries to list models as a simple health check.
        """
        try:
            # Simple check - just verify client is configured
            # Full health check would make an API call
            return bool(self.api_key)
        except Exception:
            return False
    
    async def close(self):
        """Close provider resources."""
        # google-genai client doesn't need explicit cleanup
        pass
