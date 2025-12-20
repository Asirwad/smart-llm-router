"""
API package exports.
"""

from src.api.dependencies import DBSession, ValidatedAPIKey, generate_api_key, get_api_key
from src.api.routes import router
from src.api.schemas import (
    APIKeyCreate,
    APIKeyListResponse,
    APIKeyResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    HealthResponse,
)

__all__ = [
    # Router
    "router",
    # Dependencies
    "get_api_key",
    "generate_api_key",
    "ValidatedAPIKey",
    "DBSession",
    # Schemas
    "CompletionRequest",
    "CompletionResponse",
    "APIKeyCreate",
    "APIKeyResponse",
    "APIKeyListResponse",
    "HealthResponse",
    "ErrorResponse",
]
