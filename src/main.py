"""
Smart Model Router - FastAPI Application Entry Point

This module initializes the FastAPI application with:
- Database connection management
- Health checks
- API routes (added in Phase 3)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

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


app = FastAPI(
    title="Smart Model Router",
    description="Cost-Control Smart Model Router - Routes prompts to the cheapest capable model",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "smart-model-router"}


@app.get("/")
async def root():
    """Root endpoint with basic API info."""
    return {
        "name": "Smart Model Router",
        "version": "0.1.0",
        "docs": "/docs",
    }

