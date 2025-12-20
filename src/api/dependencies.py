"""
FastAPI dependencies for authentication and database access.

Dependencies are reusable components that can be injected into routes.
They handle cross-cutting concerns like auth, sessions, and validation.
"""

import hashlib
import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.db import APIKey, get_session


async def get_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    session: AsyncSession = Depends(get_session),
) -> APIKey:
    """
    Validate the API key from the X-API-Key header.
    
    This dependency:
    1. Extracts the X-API-Key header
    2. Hashes it for secure comparison
    3. Looks up the key in the database
    4. Updates last_used_at timestamp
    5. Returns the APIKey model or raises 401/403
    
    Usage:
        @app.post("/protected")
        async def protected_route(api_key: APIKey = Depends(get_api_key)):
            # api_key is now the validated APIKey model
            ...
    """
    settings = get_settings()
    
    # Check if header is present
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Hash the provided key for comparison
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    
    # Look up in database
    result = await session.execute(
        select(APIKey).where(APIKey.key_hash == key_hash)
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if not api_key.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key has been deactivated.",
        )
    
    # Update last used timestamp (fire and forget - don't block response)
    from datetime import datetime, timezone
    api_key.last_used_at = datetime.now(timezone.utc)
    
    return api_key


def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key and its hash.
    
    Returns:
        tuple: (raw_key, key_hash)
        - raw_key: The key to give to the user (only shown once)
        - key_hash: The hash to store in the database
    
    Key format: smr_<32 random hex characters>
    Prefix 'smr_' = Smart Model Router
    """
    # Generate 32 bytes = 64 hex characters of randomness
    random_bytes = secrets.token_hex(32)
    raw_key = f"smr_{random_bytes}"
    
    # Hash for storage
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    
    return raw_key, key_hash


# Type alias for cleaner route signatures
ValidatedAPIKey = Annotated[APIKey, Depends(get_api_key)]
DBSession = Annotated[AsyncSession, Depends(get_session)]
