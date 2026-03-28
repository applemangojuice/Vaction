"""Shared test fixtures."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from vaction.db import init_db


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database with schema initialized."""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture
def populated_db(tmp_db):
    """Database with sample series/season/episode/frame/detection data."""
    conn = tmp_db

    # Insert series
    conn.execute("INSERT INTO series (id, name, root_path) VALUES (1, 'TestShow', '/fake/path')")

    # Insert season
    conn.execute("INSERT INTO season (id, series_id, number) VALUES (1, 1, 1)")

    # Insert episodes
    conn.execute("""
        INSERT INTO episode (id, season_id, number, title, file_path, width, height, duration_secs, fps_sampled, num_frames)
        VALUES (1, 1, 1, 'Pilot', '/fake/s01e01.mp4', 1920, 1080, 2700.0, 2.0, 5400)
    """)
    conn.execute("""
        INSERT INTO episode (id, season_id, number, title, file_path, width, height, duration_secs, fps_sampled, num_frames)
        VALUES (2, 1, 2, 'Second', '/fake/s01e02.mp4', 1920, 1080, 2700.0, 2.0, 5400)
    """)

    # Insert frames for episode 1
    for i in range(10):
        conn.execute(
            "INSERT INTO frame (id, episode_id, frame_index, timestamp, delta_t) VALUES (?, 1, ?, ?, 0.5)",
            (i + 1, i, i * 0.5),
        )

    # Insert frames for episode 2
    for i in range(10):
        conn.execute(
            "INSERT INTO frame (id, episode_id, frame_index, timestamp, delta_t) VALUES (?, 2, ?, ?, 0.5)",
            (i + 11, i, i * 0.5),
        )

    # Insert detections
    # Episode 1: "person" in frames 1-5, "car" in frames 3-7
    for fid in range(1, 6):
        conn.execute(
            """INSERT INTO detection (frame_id, detector, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, pixel_area)
            VALUES (?, 'yolo', 'person', 0.9, 100, 100, 200, 400, 80000)""",
            (fid,),
        )

    for fid in range(3, 8):
        conn.execute(
            """INSERT INTO detection (frame_id, detector, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, pixel_area)
            VALUES (?, 'yolo', 'car', 0.85, 500, 300, 400, 300, 120000)""",
            (fid,),
        )

    # Episode 1: "hallway" scene in frames 1-4
    for fid in range(1, 5):
        conn.execute(
            """INSERT INTO detection (frame_id, detector, label, confidence, pixel_area)
            VALUES (?, 'clip', 'hallway', 0.7, 2073600)""",
            (fid,),
        )

    # Episode 2: "person" in frames 11-15, "office" in frames 11-20
    for fid in range(11, 16):
        conn.execute(
            """INSERT INTO detection (frame_id, detector, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, pixel_area)
            VALUES (?, 'yolo', 'person', 0.88, 150, 120, 180, 380, 68400)""",
            (fid,),
        )

    for fid in range(11, 21):
        conn.execute(
            """INSERT INTO detection (frame_id, detector, label, confidence, pixel_area)
            VALUES (?, 'clip', 'office', 0.65, 2073600)""",
            (fid,),
        )

    conn.commit()
    yield conn
