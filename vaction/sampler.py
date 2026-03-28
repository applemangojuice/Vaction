"""Frame sampling from video files using ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vaction.db import insert_frame, update_episode_frame_count


def extract_frames(
    conn,
    episode_id: int,
    video_path: str,
    output_dir: Path,
    fps: float = 2.0,
    episode_tag: str = "ep",
) -> int:
    """Extract frames from a video at the given fps.

    Returns the number of frames extracted.
    """
    frame_dir = output_dir / f"episode_{episode_id}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(frame_dir / "frame_%06d.jpg")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",  # High quality JPEG
        "-y",
        pattern,
    ]

    subprocess.run(cmd, capture_output=True, check=True)

    # Count extracted frames and insert into DB
    frame_files = sorted(frame_dir.glob("frame_*.jpg"))
    delta_t = 1.0 / fps

    for idx, frame_file in enumerate(frame_files):
        timestamp = idx * delta_t
        insert_frame(
            conn,
            episode_id=episode_id,
            frame_index=idx,
            timestamp=timestamp,
            delta_t=delta_t,
            file_path=str(frame_file),
        )

    num_frames = len(frame_files)
    update_episode_frame_count(conn, episode_id, num_frames)
    conn.commit()

    return num_frames


def sample_episodes(
    conn,
    episode_ids: list[int],
    frames_dir: Path,
    fps: float = 2.0,
    progress_callback=None,
) -> dict[int, int]:
    """Sample frames for multiple episodes.

    Returns a mapping of episode_id -> frame_count.
    """
    results = {}

    for ep_id in episode_ids:
        # Get the video path (prefer working copy)
        row = conn.execute(
            "SELECT file_path, working_path FROM episode WHERE id = ?", (ep_id,)
        ).fetchone()

        if not row:
            continue

        video_path = row["working_path"] or row["file_path"]
        count = extract_frames(conn, ep_id, video_path, frames_dir, fps)
        results[ep_id] = count

        if progress_callback:
            progress_callback(ep_id, count)

    return results
