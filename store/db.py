"""DB session/engine setup. Defaults to a local SQLite file; swap
DATABASE_URL to a Postgres URL to move this to a real server later --
store/models.py needs no changes to do that.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from store.models import Base

DEFAULT_SQLITE_PATH = Path(__file__).parent.parent / "data" / "processed" / "earnings_lens.db"
DATABASE_URL = os.environ.get("EARNINGS_LENS_DB_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

_engine = None
_SessionLocal = None


def get_engine(database_url: str = DATABASE_URL):
    global _engine
    if _engine is None or str(_engine.url) != database_url:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        _engine = create_engine(database_url, connect_args=connect_args)
    return _engine


def init_db(database_url: str = DATABASE_URL):
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def get_session(database_url: str = DATABASE_URL):
    global _SessionLocal
    engine = get_engine(database_url)
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()
