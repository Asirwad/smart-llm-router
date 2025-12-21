"""
API Routes for Smart Model Router.

Defines all HTTP endpoints organized by functionality:
- /v1/complete: Main completion endpoint
- /v1/keys: API key management
- /health: Health check
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select

from src.api.dependencies import DBSession, ValidatedAPIKey, generate_api_key
from src.api.schemas import (
    APIKeyCreate,
    APIKeyListResponse,
    APIKeyResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    HealthResponse,
)
from src.config import get_settings
from src.db import APIKey, get_session

# Create router for v1 API
router = APIRouter(prefix="/v1", tags=["v1"])


# ===================
# Completion Endpoint
# ===================

@router.post(
    "/complete",
    response_model=CompletionResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid or missing API key"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Generate a completion",
    description="Routes the prompt to the most cost-effective model based on complexity.",
)
async def create_completion(
    request: CompletionRequest,
    api_key: ValidatedAPIKey,
    session: DBSession,
) -> CompletionResponse:
    """
    Main completion endpoint.
    
    Flow:
    1. Validate API key (via dependency)
    2. Check cache for existing response (semantic or exact match)
    3. Classify prompt difficulty
    4. Route to appropriate model
    5. Cache the response
    6. Log request and cost
    7. Return response with cost data
    """
    import time
    from src.config import get_settings
    from src.core import get_router
    from src.providers import get_provider_manager, ProviderError
    from src.services import (
        get_cost_calculator, 
        get_request_logger, 
        get_cache_service,
        get_semantic_cache_service,
        CachedResponse,
        SemanticCachedResponse,
    )
    
    start_time = time.perf_counter()
    settings = get_settings()
    
    # Step 1: Check cache first (semantic or exact based on config)
    if settings.use_semantic_cache:
        semantic_cache = get_semantic_cache_service()
        cached = await semantic_cache.get(request.prompt)
        if cached:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return CompletionResponse(
                response=cached.text,
                model_used=cached.model,
                difficulty_tag=cached.difficulty_tag,
                estimated_cost=cached.estimated_cost,
                estimated_savings=cached.estimated_savings,
                latency_ms=latency_ms,
                cache_hit=True,
            )
    else:
        cache_service = get_cache_service()
        cached = await cache_service.get(request.prompt)
        if cached:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return CompletionResponse(
                response=cached.text,
                model_used=cached.model,
                difficulty_tag=cached.difficulty_tag,
                estimated_cost=cached.estimated_cost,
                estimated_savings=cached.estimated_savings,
                latency_ms=latency_ms,
                cache_hit=True,
            )
    
    # Step 2: Classify the prompt
    router_agent = get_router()
    routing_decision = await router_agent.classify(
        request.prompt,
        force_tier=request.force_tier,
    )
    
    # Step 3: Call the appropriate model
    provider_manager = get_provider_manager()
    
    try:
        response, actual_tier = await provider_manager.generate(
            prompt=request.prompt,
            tier=routing_decision.tier,
            max_retries=2,
            allow_fallback=True,
        )
    except ProviderError as e:
        # Convert provider error to HTTP error
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM provider error: {e}",
        )
    
    # Step 4: Calculate costs
    calculator = get_cost_calculator()
    cost_estimate = calculator.estimate(
        model=response.model,
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
    )
    
    # Step 5: Cache the response
    if settings.use_semantic_cache:
        await semantic_cache.set(
            request.prompt,
            SemanticCachedResponse(
                text=response.text,
                model=response.model,
                difficulty_tag=actual_tier.value,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                estimated_cost=round(cost_estimate.estimated_cost, 6),
                estimated_savings=round(cost_estimate.savings, 6),
                original_prompt=request.prompt,
            ),
        )
    else:
        await cache_service.set(
            request.prompt,
            CachedResponse(
                text=response.text,
                model=response.model,
                difficulty_tag=actual_tier.value,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                estimated_cost=round(cost_estimate.estimated_cost, 6),
                estimated_savings=round(cost_estimate.savings, 6),
            ),
        )
    
    # Step 6: Log request to database
    logger = get_request_logger()
    await logger.log_request(
        session=session,
        api_key_id=api_key.id,
        prompt=request.prompt,
        response_text=response.text,
        provider_response=response,
        tier=actual_tier,
        cache_hit=False,
    )
    
    # Calculate latency (includes all processing)
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    
    return CompletionResponse(
        response=response.text,
        model_used=response.model,
        difficulty_tag=actual_tier.value,
        estimated_cost=round(cost_estimate.estimated_cost, 6),
        estimated_savings=round(cost_estimate.savings, 6),
        latency_ms=latency_ms,
        cache_hit=False,
    )


# ===================
# API Key Management
# ===================

@router.post(
    "/keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new API key",
    description="Generate a new API key. The raw key is only shown once!",
)
async def create_api_key(
    request: APIKeyCreate,
    session: DBSession,
) -> APIKeyResponse:
    """
    Create a new API key.
    
    WARNING: The raw key is only returned once. Store it securely!
    """
    # Generate key and hash
    raw_key, key_hash = generate_api_key()
    
    # Create database record
    api_key = APIKey(
        key_hash=key_hash,
        name=request.name,
        is_active=True,
    )
    session.add(api_key)
    await session.flush()  # Get the ID without committing
    
    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key=raw_key,  # Only time we return the raw key!
        is_active=api_key.is_active,
        created_at=api_key.created_at or datetime.now(timezone.utc),
        last_used_at=api_key.last_used_at,
    )


@router.get(
    "/keys",
    response_model=APIKeyListResponse,
    summary="List all API keys",
    description="List all API keys (without the raw key values).",
)
async def list_api_keys(
    session: DBSession,
) -> APIKeyListResponse:
    """List all API keys."""
    result = await session.execute(
        select(APIKey).order_by(APIKey.created_at.desc())
    )
    keys = result.scalars().all()
    
    return APIKeyListResponse(
        keys=[
            APIKeyResponse(
                id=key.id,
                name=key.name,
                key=None,  # Never expose raw key in list
                is_active=key.is_active,
                created_at=key.created_at,
                last_used_at=key.last_used_at,
            )
            for key in keys
        ],
        total=len(keys),
    )


@router.delete(
    "/keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate an API key",
    description="Deactivate an API key (soft delete).",
)
async def deactivate_api_key(
    key_id: str,
    session: DBSession,
) -> None:
    """Deactivate an API key (soft delete)."""
    from uuid import UUID
    
    try:
        uuid_key = UUID(key_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid key ID format",
        )
    
    result = await session.execute(
        select(APIKey).where(APIKey.id == uuid_key)
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    api_key.is_active = False
