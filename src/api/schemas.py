"""
Pydantic schemas for API request/response validation.

These schemas define the contract for our API endpoints,
providing automatic validation, documentation, and serialization.
"""

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ===================
# Request Schemas
# ===================

class CompletionRequest(BaseModel):
    """
    Request body for the /v1/complete endpoint.
    
    Attributes:
        prompt: The text prompt to send to the LLM
        metadata: Optional metadata for tracking (user_id, project_id, etc.)
        force_tier: Force routing to a specific tier (overrides auto-detection)
    """
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="The text prompt to process",
        examples=["Summarize the following text in 3 bullet points..."],
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional metadata for tracking purposes",
        examples=[{"user_id": "user_123", "project_id": "proj_456"}],
    )
    force_tier: Optional[Literal["simple", "medium", "complex"]] = Field(
        default=None,
        description="Force routing to a specific tier (bypasses auto-detection)",
    )


class StructuredRequest(BaseModel):
    """
    Request body for the /v1/structure endpoint.
    
    Returns guaranteed structured JSON output conforming to provided schema.
    """
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="The text prompt to process",
        examples=["Create a 3-step plan to add a factorial function"],
    )
    system_prompt: Optional[str] = Field(
        default=None,
        max_length=10000,
        description="Optional system instruction for the model",
    )
    json_schema: dict = Field(
        ...,
        description="JSON Schema that the response must conform to",
        examples=[{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }],
    )


class APIKeyCreate(BaseModel):
    """Request body for creating a new API key."""
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name for the API key",
        examples=["Production Key", "Development Key"],
    )


# ===================
# Response Schemas
# ===================

class CompletionResponse(BaseModel):
    """
    Response from the /v1/complete endpoint.
    
    Includes the LLM response and cost tracking information.
    """
    response: str = Field(
        ...,
        description="The generated response from the LLM",
    )
    model_used: str = Field(
        ...,
        description="The model that processed this request",
        examples=["granite4:350m", "gemini-2.0-flash-exp", "gemini-1.5-pro"],
    )
    difficulty_tag: Literal["simple", "medium", "complex"] = Field(
        ...,
        description="The difficulty classification of the prompt",
    )
    estimated_cost: float = Field(
        ...,
        ge=0,
        description="Estimated cost in USD for this request",
    )
    estimated_savings: float = Field(
        ...,
        ge=0,
        description="Estimated savings vs using the most expensive model",
    )
    latency_ms: int = Field(
        ...,
        ge=0,
        description="Total request processing time in milliseconds",
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether the response was served from cache",
    )


class StructuredResponse(BaseModel):
    """
    Response from the /v1/structure endpoint.
    
    Returns guaranteed structured JSON conforming to the provided schema.
    """
    data: dict | list = Field(
        ...,
        description="The structured JSON response (parsed, not string)",
    )
    model_used: str = Field(
        ...,
        description="The model that processed this request",
        examples=["gemini-2.0-flash-exp"],
    )
    estimated_cost: float = Field(
        ...,
        ge=0,
        description="Estimated cost in USD for this request",
    )
    latency_ms: int = Field(
        ...,
        ge=0,
        description="Total request processing time in milliseconds",
    )


class APIKeyResponse(BaseModel):
    """Response when creating or listing API keys."""
    id: UUID
    name: str
    key: Optional[str] = Field(
        default=None,
        description="The raw API key (only shown once at creation)",
    )
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None


class APIKeyListResponse(BaseModel):
    """Response for listing all API keys."""
    keys: list[APIKeyResponse]
    total: int


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: Literal["healthy", "degraded", "unhealthy"]
    service: str
    version: str
    database: Literal["connected", "disconnected"]
    cache: Literal["connected", "disconnected"]


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(default=None, description="Additional details")
