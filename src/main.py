"""
Smart Model Router - FastAPI Application Entry Point

This module initializes the FastAPI application with:
- Database connection management
- API routes registration
- Health checks with dependency status
- Global exception handlers
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.db import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize DB connections, cache, etc.
    - Shutdown: Clean up resources
    """
    settings = get_settings()
    
    # === Startup ===
    print("🚀 Starting Smart Model Router...")
    print(f"   Log Level: {settings.log_level}")
    print(f"   Ollama URL: {settings.ollama_base_url}")
    print(f"   Database: {settings.database_url.split('@')[-1]}")  # Hide credentials
    
    # Initialize database connection
    try:
        await init_db()
        print("   ✅ Database connected")
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        raise
    
    yield  # Application runs here
    
    # === Shutdown ===
    print("👋 Shutting down Smart Model Router...")
    await close_db()
    print("   ✅ Database connections closed")


# Create the FastAPI application
app = FastAPI(
    title="Smart Model Router",
    description="Cost-Control Smart Model Router - Routes prompts to the cheapest capable model",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ===================
# Register API Routes
# ===================

from src.api import router as v1_router

app.include_router(v1_router)


# ===================
# Global Exception Handlers
# ===================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if get_settings().log_level == "DEBUG" else None,
        },
    )


# ===================
# Root Endpoints
# ===================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration.
    
    Returns status of all dependencies.
    """
    # Check database
    db_status = "connected"
    try:
        from src.db import engine
        async with engine.connect() as conn:
            from sqlalchemy import text
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        db_status = "disconnected"
        print("   ❌ Database connection failed", e)
    
    # TODO: Check Redis in Phase 7
    cache_status = "disconnected"
    
    overall = "healthy" if db_status == "connected" else "degraded"
    
    return {
        "status": overall,
        "service": "smart-model-router",
        "version": "0.1.0",
        "database": db_status,
        "cache": cache_status,
    }


@app.get("/")
async def root():
    """Root endpoint with basic API info."""
    return {
        "name": "Smart Model Router",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "complete": "POST /v1/complete",
            "keys": "GET/POST/DELETE /v1/keys",
            "health": "GET /health",
        },
    }
