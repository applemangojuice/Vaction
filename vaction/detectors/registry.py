"""Detector registry: manages and runs all enabled detectors."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from vaction.db import insert_detection
from vaction.models import DetectionResult


@runtime_checkable
class Detector(Protocol):
    """Protocol that all detectors must implement."""

    name: str

    def detect(
        self, frame_path: str, frame_width: int, frame_height: int
    ) -> list[DetectionResult]:
        """Analyze a frame and return detections."""
        ...

    def detect_batch(
        self, frame_paths: list[str], frame_width: int, frame_height: int
    ) -> list[list[DetectionResult]]:
        """Analyze multiple frames. Default: loop over detect()."""
        ...


class DetectorRegistry:
    """Manages and orchestrates multiple detectors."""

    def __init__(self):
        self._detectors: list[Detector] = []

    def register(self, detector: Detector) -> None:
        self._detectors.append(detector)

    @property
    def detector_names(self) -> list[str]:
        return [d.name for d in self._detectors]

    def run_on_frame(
        self,
        conn,
        frame_id: int,
        frame_path: str,
        frame_width: int,
        frame_height: int,
    ) -> int:
        """Run all detectors on a single frame. Returns number of detections."""
        total = 0
        for detector in self._detectors:
            results = detector.detect(frame_path, frame_width, frame_height)
            for det in results:
                insert_detection(
                    conn,
                    frame_id=frame_id,
                    detector=det.detector,
                    label=det.label.lower(),
                    confidence=det.confidence,
                    bbox_x=det.bbox_x,
                    bbox_y=det.bbox_y,
                    bbox_w=det.bbox_w,
                    bbox_h=det.bbox_h,
                    pixel_area=det.pixel_area,
                    attributes=det.attributes or None,
                )
                total += 1
        return total

    def run_on_batch(
        self,
        conn,
        frames: list[dict],
        frame_width: int,
        frame_height: int,
    ) -> int:
        """Run all detectors on a batch of frames.

        frames: list of dicts with keys 'id', 'file_path'
        Returns total detections across all frames.
        """
        frame_paths = [f["file_path"] for f in frames]
        total = 0

        for detector in self._detectors:
            try:
                batch_results = detector.detect_batch(
                    frame_paths, frame_width, frame_height
                )
            except (NotImplementedError, AttributeError):
                # Fallback to single-frame processing
                batch_results = [
                    detector.detect(fp, frame_width, frame_height)
                    for fp in frame_paths
                ]

            for frame_info, results in zip(frames, batch_results):
                for det in results:
                    insert_detection(
                        conn,
                        frame_id=frame_info["id"],
                        detector=det.detector,
                        label=det.label.lower(),
                        confidence=det.confidence,
                        bbox_x=det.bbox_x,
                        bbox_y=det.bbox_y,
                        bbox_w=det.bbox_w,
                        bbox_h=det.bbox_h,
                        pixel_area=det.pixel_area,
                        attributes=det.attributes or None,
                    )
                    total += 1

        return total
