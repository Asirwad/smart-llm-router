"""
Database session management with async SQLAlchemy.

Provides:
- Async engine with connection pooling
- Session factory for creating database sessions
- Dependency injection helper for FastAPI
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import get_settings


def create_engine():
    """
    Create an async SQLAlchemy engine with connection pooling.
    
    Pool settings explained:
    - pool_size=5: Maintain 5 connections ready
    - max_overflow=10: Allow up to 10 extra connections under load
    - pool_pre_ping=True: Verify connections are alive before use
    - pool_recycle=3600: Recreate connections after 1 hour (prevents stale connections)
    """
    settings = get_settings()
    
    return create_async_engine(
        settings.database_url,
        echo=settings.log_level == "DEBUG",  # SQL logging in debug mode
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


# Global engine instance (created on import)
engine = create_engine()

# Session factory - creates new sessions
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevent lazy loading issues in async
    autocommit=False,
    autoflush=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    
    Usage in routes:
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...
    
    The session is automatically committed on success and rolled back on error.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions (for use outside FastAPI).
    
    Usage:
        async with get_session_context() as session:
            result = await session.execute(...)
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Initialize database connection.
    
    Called during application startup to verify connectivity.
    Note: Table creation is handled by Alembic migrations, not here.
    """
    # Import models to register them with Base.metadata
    from src.db.models import Base  # noqa: F401
    
    # Verify connection works
    async with engine.begin() as conn:
        # Just test the connection
        await conn.run_sync(lambda _: None)


async def close_db() -> None:
    """
    Close database connections.
    
    Called during application shutdown to clean up resources.
    """
    await engine.dispose()
