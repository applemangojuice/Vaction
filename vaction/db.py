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

CREATE TABLE IF NOT EXISTS theme (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS theme_tag (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_id    INTEGER NOT NULL REFERENCES theme(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    UNIQUE(theme_id, label)
);

CREATE TABLE IF NOT EXISTS clip_prompt (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt      TEXT NOT NULL UNIQUE,
    label       TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'scene',
    enabled     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS face_cluster (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id   INTEGER NOT NULL REFERENCES series(id),
    name        TEXT,
    thumbnail_frame_id INTEGER REFERENCES frame(id),
    thumbnail_bbox TEXT
);

CREATE TABLE IF NOT EXISTS face_cluster_member (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id  INTEGER NOT NULL REFERENCES face_cluster(id) ON DELETE CASCADE,
    detection_id INTEGER NOT NULL REFERENCES detection(id),
    embedding   BLOB,
    UNIQUE(cluster_id, detection_id)
);

CREATE INDEX IF NOT EXISTS idx_detection_label ON detection(label);
CREATE INDEX IF NOT EXISTS idx_detection_frame ON detection(frame_id);
CREATE INDEX IF NOT EXISTS idx_frame_episode ON frame(episode_id);
CREATE INDEX IF NOT EXISTS idx_detection_label_confidence ON detection(label, confidence);
CREATE INDEX IF NOT EXISTS idx_season_series ON season(series_id);
CREATE INDEX IF NOT EXISTS idx_episode_season ON episode(season_id);
CREATE INDEX IF NOT EXISTS idx_theme_tag_label ON theme_tag(label);
CREATE INDEX IF NOT EXISTS idx_face_cluster_series ON face_cluster(series_id);
CREATE INDEX IF NOT EXISTS idx_face_cluster_member_cluster ON face_cluster_member(cluster_id);
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


# --- Theme helpers ---

def create_theme(conn: sqlite3.Connection, name: str, description: str, labels: list[str]) -> int:
    cur = conn.execute("INSERT INTO theme (name, description) VALUES (?, ?)", (name, description))
    theme_id = cur.lastrowid
    for label in labels:
        conn.execute("INSERT OR IGNORE INTO theme_tag (theme_id, label) VALUES (?, ?)", (theme_id, label.lower()))
    conn.commit()
    return theme_id


def update_theme(conn: sqlite3.Connection, theme_id: int, name: str, description: str, labels: list[str]) -> None:
    conn.execute("UPDATE theme SET name = ?, description = ? WHERE id = ?", (name, description, theme_id))
    conn.execute("DELETE FROM theme_tag WHERE theme_id = ?", (theme_id,))
    for label in labels:
        conn.execute("INSERT OR IGNORE INTO theme_tag (theme_id, label) VALUES (?, ?)", (theme_id, label.lower()))
    conn.commit()


def delete_theme(conn: sqlite3.Connection, theme_id: int) -> None:
    conn.execute("DELETE FROM theme WHERE id = ?", (theme_id,))
    conn.commit()


def get_all_themes(conn: sqlite3.Connection) -> list[dict]:
    themes = conn.execute("SELECT * FROM theme ORDER BY name").fetchall()
    result = []
    for t in themes:
        tags = conn.execute("SELECT label FROM theme_tag WHERE theme_id = ?", (t["id"],)).fetchall()
        result.append({"id": t["id"], "name": t["name"], "description": t["description"] or "", "labels": [r["label"] for r in tags]})
    return result


def get_theme_labels(conn: sqlite3.Connection, theme_name: str) -> list[str]:
    row = conn.execute("SELECT id FROM theme WHERE name = ?", (theme_name,)).fetchone()
    if not row:
        return []
    tags = conn.execute("SELECT label FROM theme_tag WHERE theme_id = ?", (row["id"],)).fetchall()
    return [r["label"] for r in tags]


# --- CLIP prompt helpers ---

def get_clip_prompts(conn: sqlite3.Connection, enabled_only: bool = True) -> list[dict]:
    if enabled_only:
        rows = conn.execute("SELECT * FROM clip_prompt WHERE enabled = 1 ORDER BY category, label").fetchall()
    else:
        rows = conn.execute("SELECT * FROM clip_prompt ORDER BY category, label").fetchall()
    return [{"id": r["id"], "prompt": r["prompt"], "label": r["label"], "category": r["category"], "enabled": bool(r["enabled"])} for r in rows]


def add_clip_prompt(conn: sqlite3.Connection, prompt: str, label: str, category: str = "scene") -> int:
    cur = conn.execute("INSERT OR IGNORE INTO clip_prompt (prompt, label, category) VALUES (?, ?, ?)", (prompt, label.lower(), category))
    conn.commit()
    return cur.lastrowid


def delete_clip_prompt(conn: sqlite3.Connection, prompt_id: int) -> None:
    conn.execute("DELETE FROM clip_prompt WHERE id = ?", (prompt_id,))
    conn.commit()


def toggle_clip_prompt(conn: sqlite3.Connection, prompt_id: int, enabled: bool) -> None:
    conn.execute("UPDATE clip_prompt SET enabled = ? WHERE id = ?", (1 if enabled else 0, prompt_id))
    conn.commit()


def seed_default_clip_prompts(conn: sqlite3.Connection) -> None:
    """Insert default CLIP prompts if the table is empty. Uses vocabulary_presets."""
    count = conn.execute("SELECT COUNT(*) as c FROM clip_prompt").fetchone()["c"]
    if count > 0:
        return
    from vaction.vocabulary_presets import get_prompts_for_level
    # Seed with medium preset by default
    for p in get_prompts_for_level(2):
        add_clip_prompt(conn, p["prompt"], p["label"], p["category"])


def load_vocabulary_preset(conn: sqlite3.Connection, level: int) -> int:
    """Replace all CLIP prompts with those from the given preset level (1/2/3).
    Returns the number of prompts loaded."""
    from vaction.vocabulary_presets import get_prompts_for_level
    prompts = get_prompts_for_level(level)
    # Clear existing
    conn.execute("DELETE FROM clip_prompt")
    conn.commit()
    for p in prompts:
        add_clip_prompt(conn, p["prompt"], p["label"], p["category"])
    return len(prompts)


def seed_default_themes(conn: sqlite3.Connection) -> int:
    """Seed default themes from vocabulary_presets if no themes exist. Returns count created."""
    count = conn.execute("SELECT COUNT(*) as c FROM theme").fetchone()["c"]
    if count > 0:
        return 0
    from vaction.vocabulary_presets import DEFAULT_THEMES
    created = 0
    for t in DEFAULT_THEMES:
        create_theme(conn, t["name"], t["description"], t["labels"])
        created += 1
    return created
