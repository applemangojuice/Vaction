"""Video file ingestion: scanning, parsing, probing, and optional transcoding."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import NamedTuple

from vaction.config import VactionConfig
from vaction.db import get_or_create_season, get_or_create_series, insert_episode


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".ts", ".webm"}

# Patterns for parsing S01E03 style naming
EPISODE_PATTERNS = [
    re.compile(r"[Ss](\d{1,2})[Ee](\d{1,2})"),
    re.compile(r"(\d{1,2})x(\d{1,2})"),
    re.compile(r"[Ss]eason\s*(\d{1,2}).*?[Ee]pisode\s*(\d{1,2})", re.IGNORECASE),
]


class VideoInfo(NamedTuple):
    width: int
    height: int
    duration_secs: float
    codec: str


class ParsedEpisode(NamedTuple):
    season: int
    episode: int
    title: str
    file_path: Path


def probe_video(file_path: Path) -> VideoInfo:
    """Use ffprobe to extract video metadata."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise ValueError(f"No video stream found in {file_path}")

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    codec = video_stream.get("codec_name", "unknown")

    # Duration: prefer format-level, fall back to stream-level
    duration = float(
        data.get("format", {}).get("duration")
        or video_stream.get("duration", 0)
    )

    return VideoInfo(width=width, height=height, duration_secs=duration, codec=codec)


def parse_episode_from_filename(file_path: Path) -> ParsedEpisode:
    """Extract season and episode numbers from filename or path."""
    name = file_path.stem
    full_path = str(file_path)

    for pattern in EPISODE_PATTERNS:
        match = pattern.search(full_path)
        if match:
            season = int(match.group(1))
            episode = int(match.group(2))
            # Extract title: everything after the episode pattern, cleaned up
            title = name
            ep_match = pattern.search(name)
            if ep_match:
                title = name[ep_match.end():].strip(" -._")
            return ParsedEpisode(
                season=season, episode=episode, title=title, file_path=file_path
            )

    # Fallback: try to get season from directory structure
    parts = file_path.parts
    season = 1
    for part in parts:
        season_match = re.search(r"[Ss]eason\s*(\d+)", part)
        if season_match:
            season = int(season_match.group(1))
            break

    # Try to parse episode number from filename
    num_match = re.search(r"(\d+)", name)
    episode = int(num_match.group(1)) if num_match else 1

    return ParsedEpisode(
        season=season, episode=episode, title=name, file_path=file_path
    )


def scan_directory(root: Path) -> list[ParsedEpisode]:
    """Scan a directory tree for video files and parse episode info."""
    episodes = []
    for ext in VIDEO_EXTENSIONS:
        for f in root.rglob(f"*{ext}"):
            if f.is_file():
                episodes.append(parse_episode_from_filename(f))

    # Sort by season, then episode
    episodes.sort(key=lambda e: (e.season, e.episode))
    return episodes


def transcode_video(
    input_path: Path,
    output_dir: Path,
    resolution: int = 720,
    crf: int = 23,
) -> Path:
    """Transcode video to a lower-resolution working copy.

    Uses ffmpeg with VideoToolbox on macOS for hardware acceleration.
    Falls back to libx264 if VideoToolbox is unavailable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_working.mp4"

    if output_path.exists():
        return output_path

    # Scale to target height, maintain aspect ratio (even dimensions)
    scale_filter = f"scale=-2:{resolution}"

    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-vf", scale_filter,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "128k",
        "-y",
        str(output_path),
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def ingest_directory(
    conn,
    root: Path,
    series_name: str,
    config: VactionConfig,
) -> list[int]:
    """Scan a directory, register all episodes, and return episode IDs."""
    series_id = get_or_create_series(conn, series_name, str(root))
    parsed = scan_directory(root)
    episode_ids = []

    for ep in parsed:
        season_id = get_or_create_season(conn, series_id, ep.season)
        info = probe_video(ep.file_path)

        working_path = None
        if config.transcode_resolution:
            working_dir = config.data_dir / "working_copies" / series_name
            working = transcode_video(
                ep.file_path, working_dir,
                resolution=config.transcode_resolution,
                crf=config.transcode_crf,
            )
            working_path = str(working)

        ep_id = insert_episode(
            conn,
            season_id=season_id,
            number=ep.episode,
            title=ep.title,
            file_path=str(ep.file_path),
            working_path=working_path,
            width=info.width,
            height=info.height,
            duration_secs=info.duration_secs,
            fps_sampled=config.default_fps,
        )
        episode_ids.append(ep_id)

    return episode_ids
