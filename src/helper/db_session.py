"""
Database session management utilities.

Provides functions for creating SQLAlchemy engines and session factories
for SQLite database connections.

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from __future__ import annotations
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from src.helper.config import AppConfig


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_engine(config: AppConfig) -> Engine:
    database_path = Path(config.ingestion.canonical_db_path)
    ensure_parent_directory(database_path)
    connection_string = f"sqlite:///{database_path}"
    return create_engine(connection_string, connect_args={"check_same_thread": False})


def get_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
