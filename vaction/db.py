"""SQLite database management for Vaction."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS series (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    root_path   TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS season (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id   INTEGER NOT NULL REFERENCES series(id),
    number      INTEGER NOT NULL,
    UNIQUE(series_id, number)
);

CREATE TABLE IF NOT EXISTS episode (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    season_id       INTEGER NOT NULL REFERENCES season(id),
    number          INTEGER NOT NULL,
    title           TEXT,
    file_path       TEXT NOT NULL UNIQUE,
    working_path    TEXT,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    duration_secs   REAL NOT NULL,
    fps_sampled     REAL NOT NULL,
    num_frames      INTEGER NOT NULL DEFAULT 0,
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(season_id, number)
);

CREATE TABLE IF NOT EXISTS frame (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id  INTEGER NOT NULL REFERENCES episode(id),
    frame_index INTEGER NOT NULL,
    timestamp   REAL NOT NULL,
    delta_t     REAL NOT NULL,
    file_path   TEXT,
    UNIQUE(episode_id, frame_index)
);

CREATE TABLE IF NOT EXISTS detection (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id    INTEGER NOT NULL REFERENCES frame(id),
    detector    TEXT NOT NULL,
    label       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    bbox_x      INTEGER,
    bbox_y      INTEGER,
    bbox_w      INTEGER,
    bbox_h      INTEGER,
    pixel_area  INTEGER NOT NULL,
    attributes  TEXT
);

CREATE TABLE IF NOT EXISTS character (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id   INTEGER NOT NULL REFERENCES series(id),
    name        TEXT NOT NULL,
    UNIQUE(series_id, name)
);

CREATE TABLE IF NOT EXISTS character_embedding (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    character_id    INTEGER NOT NULL REFERENCES character(id),
    embedding       BLOB NOT NULL,
    source_frame_id INTEGER REFERENCES frame(id)
);

CREATE TABLE IF NOT EXISTS alias (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    term        TEXT NOT NULL,
    label       TEXT NOT NULL,
    UNIQUE(term, label)
);

CREATE INDEX IF NOT EXISTS idx_detection_label ON detection(label);
CREATE INDEX IF NOT EXISTS idx_detection_frame ON detection(frame_id);
CREATE INDEX IF NOT EXISTS idx_frame_episode ON frame(episode_id);
CREATE INDEX IF NOT EXISTS idx_detection_label_confidence ON detection(label, confidence);
CREATE INDEX IF NOT EXISTS idx_season_series ON season(series_id);
CREATE INDEX IF NOT EXISTS idx_episode_season ON episode(season_id);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Create a database connection with proper settings."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the database schema."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


@contextmanager
def db_session(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database sessions with auto-commit."""
    conn = init_db(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --- CRUD helpers ---

def get_or_create_series(conn: sqlite3.Connection, name: str, root_path: str) -> int:
    row = conn.execute("SELECT id FROM series WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO series (name, root_path) VALUES (?, ?)", (name, root_path))
    conn.commit()
    return cur.lastrowid


def get_or_create_season(conn: sqlite3.Connection, series_id: int, number: int) -> int:
    row = conn.execute(
        "SELECT id FROM season WHERE series_id = ? AND number = ?",
        (series_id, number),
    ).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO season (series_id, number) VALUES (?, ?)",
        (series_id, number),
    )
    conn.commit()
    return cur.lastrowid


def insert_episode(
    conn: sqlite3.Connection,
    season_id: int,
    number: int,
    title: str,
    file_path: str,
    working_path: str | None,
    width: int,
    height: int,
    duration_secs: float,
    fps_sampled: float,
) -> int:
    cur = conn.execute(
        """INSERT OR REPLACE INTO episode
        (season_id, number, title, file_path, working_path, width, height, duration_secs, fps_sampled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (season_id, number, title, file_path, working_path, width, height, duration_secs, fps_sampled),
    )
    conn.commit()
    return cur.lastrowid


def insert_frame(
    conn: sqlite3.Connection,
    episode_id: int,
    frame_index: int,
    timestamp: float,
    delta_t: float,
    file_path: str | None = None,
) -> int:
    cur = conn.execute(
        """INSERT OR REPLACE INTO frame (episode_id, frame_index, timestamp, delta_t, file_path)
        VALUES (?, ?, ?, ?, ?)""",
        (episode_id, frame_index, timestamp, delta_t, file_path),
    )
    return cur.lastrowid


def insert_detection(
    conn: sqlite3.Connection,
    frame_id: int,
    detector: str,
    label: str,
    confidence: float,
    bbox_x: int | None,
    bbox_y: int | None,
    bbox_w: int | None,
    bbox_h: int | None,
    pixel_area: int,
    attributes: dict | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO detection
        (frame_id, detector, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, pixel_area, attributes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            frame_id, detector, label, confidence,
            bbox_x, bbox_y, bbox_w, bbox_h, pixel_area,
            json.dumps(attributes) if attributes else None,
        ),
    )
    return cur.lastrowid


def insert_alias(conn: sqlite3.Connection, term: str, label: str) -> None:
    """Insert a search alias (e.g., 'vehicle' -> 'car')."""
    conn.execute(
        "INSERT OR IGNORE INTO alias (term, label) VALUES (?, ?)",
        (term.lower(), label.lower()),
    )
    conn.commit()


def get_aliases(conn: sqlite3.Connection, term: str) -> list[str]:
    """Get all labels that a search term maps to."""
    rows = conn.execute(
        "SELECT label FROM alias WHERE term = ?", (term.lower(),)
    ).fetchall()
    return [row["label"] for row in rows]


def update_episode_frame_count(conn: sqlite3.Connection, episode_id: int, count: int) -> None:
    conn.execute("UPDATE episode SET num_frames = ? WHERE id = ?", (count, episode_id))
    conn.commit()
