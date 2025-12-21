"""
Redis Cache Service.

Provides exact-match prompt caching to avoid redundant LLM calls:
- Hash-based key generation for prompt lookup
- TTL-based expiration
- JSON serialization for cached responses
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Optional

import redis.asyncio as redis

from src.config import get_settings


@dataclass
class CachedResponse:
    """
    Cached response structure.
    
    Stores all data needed to return a cached response
    without calling the LLM again.
    """
    text: str
    model: str
    difficulty_tag: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    estimated_savings: float
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> "CachedResponse":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class CacheService:
    """
    Redis-based caching for LLM responses.
    
    Key format: smr:cache:{prompt_hash}
    Value: JSON serialized CachedResponse
    TTL: Configurable (default 1 hour)
    """
    
    PREFIX = "smr:cache:"
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Initialize cache service.
        
        Args:
            redis_url: Redis connection URL (defaults to settings)
            ttl_seconds: Time-to-live for cached entries (defaults to settings)
        """
        settings = get_settings()
        self.redis_url = redis_url or settings.redis_url
        self.ttl_seconds = ttl_seconds or settings.cache_ttl_seconds
        self._client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Establish Redis connection."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
    
    async def disconnect(self):
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> bool:
        """Check if Redis is available."""
        try:
            await self.connect()
            await self._client.ping()
            return True
        except Exception:
            return False
    
    def _hash_prompt(self, prompt: str) -> str:
        """
        Generate cache key from prompt.
        
        Uses SHA-256 for consistent, collision-resistant hashing.
        """
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def _make_key(self, prompt_hash: str) -> str:
        """Generate full Redis key with prefix."""
        return f"{self.PREFIX}{prompt_hash}"
    
    async def get(self, prompt: str) -> Optional[CachedResponse]:
        """
        Look up cached response for a prompt.
        
        Args:
            prompt: The exact prompt text
            
        Returns:
            CachedResponse if found, None otherwise
        """
        try:
            await self.connect()
            prompt_hash = self._hash_prompt(prompt)
            key = self._make_key(prompt_hash)
            
            cached = await self._client.get(key)
            if cached:
                return CachedResponse.from_json(cached)
            return None
        except Exception as e:
            # Cache failures should not break the app
            print(f"⚠️ Cache get error: {e}")
            return None
    
    async def set(
        self,
        prompt: str,
        response: CachedResponse,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache a response.
        
        Args:
            prompt: The exact prompt text
            response: Response to cache
            ttl: Optional TTL override (seconds)
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            await self.connect()
            prompt_hash = self._hash_prompt(prompt)
            key = self._make_key(prompt_hash)
            
            await self._client.setex(
                key,
                ttl or self.ttl_seconds,
                response.to_json(),
            )
            return True
        except Exception as e:
            # Cache failures should not break the app
            print(f"⚠️ Cache set error: {e}")
            return False
    
    async def delete(self, prompt: str) -> bool:
        """
        Delete a cached response.
        
        Args:
            prompt: The exact prompt text
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            await self.connect()
            prompt_hash = self._hash_prompt(prompt)
            key = self._make_key(prompt_hash)
            
            await self._client.delete(key)
            return True
        except Exception as e:
            print(f"⚠️ Cache delete error: {e}")
            return False
    
    async def clear_all(self) -> int:
        """
        Clear all cached responses.
        
        Returns:
            Number of keys deleted
        """
        try:
            await self.connect()
            pattern = f"{self.PREFIX}*"
            keys = await self._client.keys(pattern)
            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as e:
            print(f"⚠️ Cache clear error: {e}")
            return 0
    
    async def stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats (key_count, memory_usage, etc.)
        """
        try:
            await self.connect()
            pattern = f"{self.PREFIX}*"
            keys = await self._client.keys(pattern)
            
            info = await self._client.info("memory")
            
            return {
                "key_count": len(keys),
                "memory_used_bytes": info.get("used_memory", 0),
                "memory_used_human": info.get("used_memory_human", "0B"),
                "ttl_seconds": self.ttl_seconds,
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_cache: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get the singleton cache service."""
    global _cache
    if _cache is None:
        _cache = CacheService()
    return _cache
