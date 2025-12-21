"""
Embeddings Service.

Generates text embeddings using Ollama's embedding models.
Used for semantic similarity caching.
"""

from typing import Optional
import httpx

from src.config import get_settings


class EmbeddingsService:
    """
    Generates embeddings using Ollama.
    
    Uses the nomic-embed-text model by default.
    """
    
    # Default embedding model for Ollama
    DEFAULT_MODEL = "nomic-embed-text"
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize embeddings service.
        
        Args:
            base_url: Ollama API URL (defaults to settings)
            model: Embedding model to use (defaults to nomic-embed-text)
        """
        settings = get_settings()
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or self.DEFAULT_MODEL
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60)
        return self._client
    
    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        client = await self._get_client()
        
        response = await client.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": text,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        # Ollama returns embeddings in data["embeddings"][0]
        embeddings = data.get("embeddings", [[]])[0]
        return embeddings
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Ollama's embed API supports batch input
        client = await self._get_client()
        
        response = await client.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        return data.get("embeddings", [])
    
    async def health_check(self) -> bool:
        """Check if embedding model is available."""
        try:
            # Try to embed a simple text
            embedding = await self.embed("test")
            return len(embedding) > 0
        except Exception:
            return False
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance
_embeddings: Optional[EmbeddingsService] = None


def get_embeddings_service() -> EmbeddingsService:
    """Get the singleton embeddings service."""
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingsService()
    return _embeddings
