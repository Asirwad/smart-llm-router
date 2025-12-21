"""
Semantic Cache Service.

Provides similarity-based caching using embeddings and Redis vector search.
Finds similar prompts (not just exact matches) to improve cache hit rate.
"""

import json
import hashlib
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import redis.asyncio as redis

from src.config import get_settings
from src.services.embeddings import get_embeddings_service


@dataclass
class SemanticCachedResponse:
    """
    Cached response with embedding metadata.
    """
    text: str
    model: str
    difficulty_tag: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    estimated_savings: float
    original_prompt: str  # The original prompt that generated this response
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> "SemanticCachedResponse":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)


class SemanticCacheService:
    """
    Semantic caching using Redis vector search.
    
    Stores responses with their prompt embeddings.
    On lookup, finds similar prompts using cosine similarity.
    """
    
    INDEX_NAME = "smr_semantic_idx"
    PREFIX = "smr:semantic:"
    EMBEDDING_DIM = 768  # nomic-embed-text dimension
    SIMILARITY_THRESHOLD = 0.92  # Minimum similarity for cache hit
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        similarity_threshold: float = 0.92,
    ):
        """
        Initialize semantic cache.
        
        Args:
            redis_url: Redis connection URL
            ttl_seconds: TTL for cached entries
            similarity_threshold: Minimum cosine similarity for match
        """
        settings = get_settings()
        self.redis_url = redis_url or settings.redis_url
        self.ttl_seconds = ttl_seconds or settings.cache_ttl_seconds
        self.similarity_threshold = similarity_threshold
        self._client: Optional[redis.Redis] = None
        self._index_created = False
        self._embeddings = get_embeddings_service()
    
    async def connect(self):
        """Establish Redis connection."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
    
    async def _ensure_index(self):
        """Create vector search index if not exists."""
        if self._index_created:
            return
        
        await self.connect()
        
        try:
            # Check if index exists
            await self._client.execute_command("FT.INFO", self.INDEX_NAME)
            self._index_created = True
        except redis.ResponseError:
            # Create the index
            await self._client.execute_command(
                "FT.CREATE", self.INDEX_NAME,
                "ON", "HASH",
                "PREFIX", "1", self.PREFIX,
                "SCHEMA",
                "embedding", "VECTOR", "HNSW", "6",
                    "TYPE", "FLOAT32",
                    "DIM", str(self.EMBEDDING_DIM),
                    "DISTANCE_METRIC", "COSINE",
                "response_json", "TEXT",
                "prompt_hash", "TAG",
            )
            self._index_created = True
    
    def _hash_prompt(self, prompt: str) -> str:
        """Generate hash for prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def _embedding_to_bytes(self, embedding: list[float]) -> bytes:
        """Convert embedding to bytes for Redis."""
        return np.array(embedding, dtype=np.float32).tobytes()
    
    async def get(self, prompt: str) -> Optional[SemanticCachedResponse]:
        """
        Find cached response for similar prompt.
        
        Args:
            prompt: User prompt to search for
            
        Returns:
            SemanticCachedResponse if similar prompt found, None otherwise
        """
        try:
            await self._ensure_index()
            
            # Generate embedding for query
            query_embedding = await self._embeddings.embed(prompt)
            
            if not query_embedding:
                return None
            
            # Vector similarity search
            query_bytes = self._embedding_to_bytes(query_embedding)
            
            # Use FT.SEARCH with KNN
            result = await self._client.execute_command(
                "FT.SEARCH", self.INDEX_NAME,
                f"*=>[KNN 1 @embedding $vec AS score]",
                "PARAMS", "2", "vec", query_bytes,
                "SORTBY", "score",
                "RETURN", "2", "response_json", "score",
                "DIALECT", "2",
            )
            
            # Parse result: [total_count, key1, [field1, value1, ...], ...]
            if result[0] == 0:
                return None
            
            # Extract score and response
            fields = result[2]  # [field_name, value, ...]
            field_dict = dict(zip(fields[::2], fields[1::2]))
            
            score = float(field_dict.get("score", 0))
            similarity = 1 - score  # Cosine distance to similarity
            
            if similarity < self.similarity_threshold:
                return None
            
            response_json = field_dict.get("response_json")
            if response_json:
                return SemanticCachedResponse.from_json(response_json)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Semantic cache get error: {e}")
            return None
    
    async def set(
        self,
        prompt: str,
        response: SemanticCachedResponse,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache a response with its embedding.
        
        Args:
            prompt: Original prompt
            response: Response to cache
            ttl: Optional TTL override
            
        Returns:
            True if cached successfully
        """
        try:
            await self._ensure_index()
            
            # Generate embedding
            embedding = await self._embeddings.embed(prompt)
            
            if not embedding:
                return False
            
            # Store in Redis as hash
            prompt_hash = self._hash_prompt(prompt)
            key = f"{self.PREFIX}{prompt_hash}"
            
            await self._client.hset(
                key,
                mapping={
                    "embedding": self._embedding_to_bytes(embedding),
                    "response_json": response.to_json(),
                    "prompt_hash": prompt_hash,
                },
            )
            
            # Set TTL
            await self._client.expire(key, ttl or self.ttl_seconds)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Semantic cache set error: {e}")
            return False
    
    async def clear_all(self) -> int:
        """Clear all semantic cache entries."""
        try:
            await self.connect()
            keys = await self._client.keys(f"{self.PREFIX}*")
            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as e:
            print(f"⚠️ Semantic cache clear error: {e}")
            return 0


# Singleton
_semantic_cache: Optional[SemanticCacheService] = None


def get_semantic_cache_service() -> SemanticCacheService:
    """Get the singleton semantic cache service."""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCacheService()
    return _semantic_cache
