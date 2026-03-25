"""Data models for Vaction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Series:
    id: int | None = None
    name: str = ""
    root_path: str = ""


@dataclass
class Season:
    id: int | None = None
    series_id: int = 0
    number: int = 0


@dataclass
class Episode:
    id: int | None = None
    season_id: int = 0
    number: int = 0
    title: str = ""
    file_path: str = ""
    working_path: str | None = None
    width: int = 0
    height: int = 0
    duration_secs: float = 0.0
    fps_sampled: float = 2.0
    num_frames: int = 0


@dataclass
class Frame:
    id: int | None = None
    episode_id: int = 0
    frame_index: int = 0
    timestamp: float = 0.0
    delta_t: float = 0.5
    file_path: str | None = None


@dataclass
class DetectionResult:
    """Result from a single detector for one region in a frame."""
    detector: str = ""
    label: str = ""
    confidence: float = 0.0
    bbox_x: int | None = None
    bbox_y: int | None = None
    bbox_w: int | None = None
    bbox_h: int | None = None
    pixel_area: int = 0
    attributes: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result of a search query for one episode."""
    series_name: str = ""
    season_number: int = 0
    episode_number: int = 0
    episode_title: str = ""
    query: str = ""
    pixel_time: float = 0.0  # pixel-seconds
    pixel_minutes: float = 0.0
    episode_share: float = 0.0  # 0.0 to 1.0
    episode_share_pct: float = 0.0  # 0.0 to 100.0
    total_pixel_budget: float = 0.0
    strongest_windows: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Character:
    id: int | None = None
    series_id: int = 0
    name: str = ""
