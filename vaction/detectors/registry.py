"""Detector registry: manages and runs all enabled detectors."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, runtime_checkable

from vaction.db import insert_detection, insert_detections_batch
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

    def _run_detector_on_batch(
        self,
        detector: Detector,
        frame_paths: list[str],
        frame_width: int,
        frame_height: int,
    ) -> list[list[DetectionResult]]:
        """Run a single detector on a batch, with fallback to per-frame."""
        try:
            return detector.detect_batch(frame_paths, frame_width, frame_height)
        except (NotImplementedError, AttributeError):
            return [
                detector.detect(fp, frame_width, frame_height)
                for fp in frame_paths
            ]

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

        Detectors run concurrently (they use separate models/GPU streams).
        DB inserts are batched with executemany for throughput.
        """
        frame_paths = [f["file_path"] for f in frames]

        # Run detectors concurrently — each uses its own model/GPU stream
        all_detector_results: list[list[list[DetectionResult]]] = []
        if len(self._detectors) > 1:
            with ThreadPoolExecutor(max_workers=len(self._detectors)) as pool:
                futures = [
                    pool.submit(
                        self._run_detector_on_batch,
                        detector, frame_paths, frame_width, frame_height,
                    )
                    for detector in self._detectors
                ]
                all_detector_results = [f.result() for f in futures]
        else:
            for detector in self._detectors:
                all_detector_results.append(
                    self._run_detector_on_batch(
                        detector, frame_paths, frame_width, frame_height
                    )
                )

        # Collect all detections into a flat list for batch insert
        detection_rows: list[dict] = []
        for batch_results in all_detector_results:
            for frame_info, results in zip(frames, batch_results):
                for det in results:
                    detection_rows.append({
                        "frame_id": frame_info["id"],
                        "detector": det.detector,
                        "label": det.label.lower(),
                        "confidence": det.confidence,
                        "bbox_x": det.bbox_x,
                        "bbox_y": det.bbox_y,
                        "bbox_w": det.bbox_w,
                        "bbox_h": det.bbox_h,
                        "pixel_area": det.pixel_area,
                        "attributes": det.attributes or None,
                    })

        total = insert_detections_batch(conn, detection_rows)
        return total
