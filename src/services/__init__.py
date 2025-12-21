"""
Services package.

Business logic and cross-cutting concerns.
"""

from src.services.cache import (
    CachedResponse,
    CacheService,
    get_cache_service,
)
from src.services.cost import (
    CostCalculator,
    CostEstimate,
    ModelPricing,
    RequestLogger,
    get_cost_calculator,
    get_request_logger,
    MODEL_PRICING,
    BASELINE_MODEL,
)
from src.services.embeddings import (
    EmbeddingsService,
    get_embeddings_service,
)
from src.services.semantic_cache import (
    SemanticCacheService,
    SemanticCachedResponse,
    get_semantic_cache_service,
)

__all__ = [
    # Cache
    "CacheService",
    "CachedResponse",
    "get_cache_service",
    # Semantic Cache
    "SemanticCacheService",
    "SemanticCachedResponse",
    "get_semantic_cache_service",
    # Embeddings
    "EmbeddingsService",
    "get_embeddings_service",
    # Cost
    "CostCalculator",
    "CostEstimate",
    "ModelPricing",
    "RequestLogger",
    "get_cost_calculator",
    "get_request_logger",
    "MODEL_PRICING",
    "BASELINE_MODEL",
]

