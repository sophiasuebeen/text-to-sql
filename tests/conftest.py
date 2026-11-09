import sqlite3
from pathlib import Path

import pytest

from text_to_sql.sql.executor import Limits


@pytest.fixture
def make_db(tmp_path):
    """make_db("CREATE ...; INSERT ...") -> path to a fresh SQLite file."""
    counter = {"n": 0}

    def _make(script: str) -> Path:
        counter["n"] += 1
        path = tmp_path / f"db{counter['n']}.sqlite"
        conn = sqlite3.connect(path)
        conn.executescript(script)
        conn.commit()
        conn.close()
        return path

    return _make


@pytest.fixture
def people_db(make_db):
    return make_db(
        """
        CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, score REAL);
        INSERT INTO people VALUES (1, 'Ann', 30, 1.5), (2, 'bob', 25, 2.0),
                                  (3, 'Cid', 30, NULL), (4, 'Dee', 41, 2.0);
        """
    )


@pytest.fixture
def fast_limits():
    return Limits(wall_seconds=3.0)
