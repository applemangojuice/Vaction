"""Configuration management for Vaction."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


def _default_data_dir() -> Path:
    return Path.home() / ".vaction"


def _default_device() -> str:
    if platform.system() == "Darwin":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


class VactionConfig(BaseSettings):
    """Global configuration for Vaction."""

    model_config = {"env_prefix": "VACTION_"}

    # Paths
    data_dir: Path = Field(default_factory=_default_data_dir)
    db_name: str = "vaction.db"
    frame_output_dir: str = "frames"

    # Frame sampling
    default_fps: float = 2.0

    # Transcode
    transcode_resolution: int | None = None  # e.g. 720 for 720p, None = no transcode
    transcode_crf: int = 23

    # Detection
    yolo_model: str = "yolov8m.pt"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    confidence_threshold: float = 0.25
    face_distance_threshold: float = 0.6
    device: str = Field(default_factory=_default_device)

    # Processing
    batch_size: int = 16
    delete_frames_after_detection: bool = False

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_name

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / self.frame_output_dir

    def ensure_dirs(self) -> None:
        """Create required directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)


def get_config(**overrides) -> VactionConfig:
    """Get configuration with optional overrides."""
    return VactionConfig(**overrides)
