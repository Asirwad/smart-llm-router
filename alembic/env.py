"""
Alembic environment configuration for async SQLAlchemy.

This file is executed when Alembic commands run. It configures:
- Database connection (from environment)
- Metadata for autogenerate
- Async migration support
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool

# Import our models so Alembic can detect them
from src.db.models import Base
from src.config import get_settings

# Alembic Config object
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata

# Get database URL from our app settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    This generates SQL scripts without connecting to the database.
    Useful for generating migration scripts for DBA review.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations with the given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in 'online' mode with async engine.
    
    Creates an async engine and runs migrations within a connection.
    """
    from sqlalchemy.ext.asyncio import create_async_engine
    
    # Create engine directly with our settings URL
    connectable = create_async_engine(
        settings.database_url,
        poolclass=pool.NullPool,  # Don't pool for migrations
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    
    Uses asyncio to run async migrations.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
