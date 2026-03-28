"""Face detection and character identification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vaction.models import DetectionResult


class FaceDetector:
    """Face detection and recognition using insightface."""

    name = "face"

    def __init__(
        self,
        device: str = "cpu",
        distance_threshold: float = 0.6,
    ):
        self.distance_threshold = distance_threshold
        self._character_embeddings: dict[str, list[np.ndarray]] = {}
        self._app = None
        self._device = device

    def _ensure_model(self):
        """Lazy-load the insightface model."""
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0 if self._device == "cpu" else 0, det_size=(640, 640))
        except ImportError:
            raise ImportError(
                "insightface is required for face detection. "
                "Install it with: pip install insightface onnxruntime"
            )

    def load_character_embeddings(self, conn, series_id: int) -> None:
        """Load enrolled character embeddings from the database."""
        rows = conn.execute(
            """SELECT c.name, ce.embedding
            FROM character_embedding ce
            JOIN character c ON ce.character_id = c.id
            WHERE c.series_id = ?""",
            (series_id,),
        ).fetchall()

        self._character_embeddings.clear()
        for row in rows:
            name = row["name"]
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            self._character_embeddings.setdefault(name, []).append(embedding)

    def _identify_face(self, embedding: np.ndarray) -> tuple[str, float]:
        """Match a face embedding against enrolled characters.

        Returns (name, distance). If no match, returns ("unknown_face", 1.0).
        """
        best_name = "unknown_face"
        best_distance = 1.0

        for name, ref_embeddings in self._character_embeddings.items():
            for ref_emb in ref_embeddings:
                # Cosine distance
                sim = np.dot(embedding, ref_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(ref_emb) + 1e-8
                )
                distance = 1.0 - sim
                if distance < best_distance:
                    best_distance = distance
                    best_name = name

        if best_distance > self.distance_threshold:
            return "unknown_face", best_distance

        return best_name, best_distance

    def detect(
        self, frame_path: str, frame_width: int, frame_height: int
    ) -> list[DetectionResult]:
        import cv2

        self._ensure_model()

        img = cv2.imread(frame_path)
        if img is None:
            return []

        faces = self._app.get(img)
        detections = []

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            bbox_w = max(0, x2 - x1)
            bbox_h = max(0, y2 - y1)

            # Identify if we have enrolled characters
            label = "face"
            confidence = float(face.det_score)

            if face.embedding is not None and self._character_embeddings:
                name, distance = self._identify_face(face.embedding)
                label = name
                confidence = max(0.0, 1.0 - distance)

            detections.append(DetectionResult(
                detector=self.name,
                label=label,
                confidence=confidence,
                bbox_x=int(x1),
                bbox_y=int(y1),
                bbox_w=bbox_w,
                bbox_h=bbox_h,
                pixel_area=bbox_w * bbox_h,
                attributes={"det_score": float(face.det_score)},
            ))

        return detections

    def detect_batch(
        self, frame_paths: list[str], frame_width: int, frame_height: int
    ) -> list[list[DetectionResult]]:
        # insightface doesn't natively batch, so we loop
        return [
            self.detect(fp, frame_width, frame_height) for fp in frame_paths
        ]


def enroll_character(
    conn,
    series_id: int,
    character_name: str,
    image_paths: list[str],
    device: str = "cpu",
) -> int:
    """Enroll a character by computing face embeddings from reference images.

    Returns the number of embeddings stored.
    """
    import cv2

    detector = FaceDetector(device=device)
    detector._ensure_model()

    # Get or create character
    row = conn.execute(
        "SELECT id FROM character WHERE series_id = ? AND name = ?",
        (series_id, character_name),
    ).fetchone()

    if row:
        char_id = row["id"]
    else:
        cur = conn.execute(
            "INSERT INTO character (series_id, name) VALUES (?, ?)",
            (series_id, character_name),
        )
        char_id = cur.lastrowid

    count = 0
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = detector._app.get(img)
        if not faces:
            continue

        # Use the largest face in the image
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        if largest.embedding is not None:
            embedding_bytes = largest.embedding.astype(np.float32).tobytes()
            conn.execute(
                "INSERT INTO character_embedding (character_id, embedding) VALUES (?, ?)",
                (char_id, embedding_bytes),
            )
            count += 1

    conn.commit()
    return count
