"""
Configuration management using Pydantic Settings.

This module centralizes all configuration from environment variables
with type safety and validation.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===================
    # Database
    # ===================
    database_url: str = "postgresql+asyncpg://router_user:router_password@localhost:5432/router_db"

    # ===================
    # Redis
    # ===================
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # ===================
    # Ollama (Local LLM)
    # ===================
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "granite4:350m"

    # ===================
    # Google Gemini
    # ===================
    google_api_key: str = ""
    gemini_flash_model: str = "gemini-2.0-flash-exp"
    gemini_pro_model: str = "gemini-2.5-pro"

    # ===================
    # API Security
    # ===================
    api_key_header: str = "X-API-Key"

    # ===================
    # Routing Thresholds
    # ===================
    simple_token_threshold: int = 50   # Below this = simple (very short prompts)
    complex_token_threshold: int = 500  # Above this = complex (very long prompts)
    use_llm_router: bool = True  # Use LLM-based classification (vs rule-based)
    use_semantic_cache: bool = True  # Use semantic similarity cache (vs exact match)

    # ===================
    # Logging
    # ===================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once,
    improving performance and ensuring consistency.
    """
    return Settings()
