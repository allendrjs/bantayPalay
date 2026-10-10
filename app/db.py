"""Database connection.

Defaults to a local SQLite file so it runs anywhere with no setup.
For MySQL (e.g. on Hostinger) set DATABASE_URL, for example:
    mysql+pymysql://user:password@host:3306/elnino_watch
and `pip install pymysql`.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = f"sqlite:///{ROOT / 'data' / 'elnino.db'}"

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(os.environ.get("DATABASE_URL", DEFAULT_URL))
    return _engine


def reset_engine() -> None:
    """Used by tests after changing DATABASE_URL."""
    global _engine
    _engine = None


def read_table(name: str) -> pd.DataFrame:
    return pd.read_sql_table(name, get_engine())
