"""
SQLAlchemy ORM Models for Smart Model Router.

Defines the database schema using SQLAlchemy 2.0 declarative style
with full async support.
"""

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """
    Base class for all ORM models.
    
    Using DeclarativeBase (SQLAlchemy 2.0 style) instead of
    declarative_base() for better type hints and IDE support.
    """
    pass


class DifficultyTag(str, enum.Enum):
    """Enum for prompt difficulty classification."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class APIKey(Base):
    """
    API Key model for authentication.
    
    Design decisions:
    - Store hash of key, never the raw key (security)
    - Track last_used_at for audit/cleanup purposes
    - is_active allows disabling without deleting (audit trail)
    """
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    key_hash: Mapped[str] = mapped_column(
        String(64),  # SHA-256 produces 64 hex characters
        unique=True,
        nullable=False,
        index=True,  # Fast lookups on auth
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationship to request logs
    requests: Mapped[list["RequestLog"]] = relationship(
        back_populates="api_key",
        lazy="selectin",  # Async-compatible eager loading
    )

    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name={self.name}, active={self.is_active})>"


class RequestLog(Base):
    """
    Request logging model for cost tracking and analytics.
    
    Design decisions:
    - prompt_hash instead of raw prompt (privacy)
    - Store both estimated_cost and baseline_cost for savings calculation
    - latency_ms for performance monitoring
    - response_hash optional for cache validation
    """
    __tablename__ = "requests_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,  # Common query pattern: filter by time range
    )
    
    # Foreign key to API key (optional - system requests may not have one)
    api_key_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    
    # Request metadata
    prompt_hash: Mapped[str] = mapped_column(
        String(64),  # SHA-256 hash
        nullable=False,
    )
    prompt_length: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    
    # Routing decision
    difficulty_tag: Mapped[DifficultyTag] = mapped_column(
        Enum(DifficultyTag, name="difficulty_tag_enum"),
        nullable=False,
        index=True,
    )
    model_used: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    
    # Token counts
    input_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    output_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    
    # Cost tracking
    estimated_cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    baseline_cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Cost if GPT-4o/Gemini Pro was used",
    )
    cost_saved: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    
    # Performance
    latency_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    
    # Cache tracking
    cache_hit: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    
    # Optional: store response hash for cache validation
    response_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )
    
    # Optional: store error if request failed
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # Relationship back to API key
    api_key: Mapped[Optional["APIKey"]] = relationship(
        back_populates="requests",
    )

    def __repr__(self) -> str:
        return f"<RequestLog(id={self.id}, model={self.model_used}, cost={self.estimated_cost})>"
