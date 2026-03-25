"""YOLOv8 object detection wrapper."""

from __future__ import annotations

from vaction.models import DetectionResult


class YOLODetector:
    """Object detection using YOLOv8 via ultralytics."""

    name = "yolo"

    def __init__(self, model_name: str = "yolov8m.pt", device: str = "cpu", confidence: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.device = device
        self.confidence = confidence

    def detect(
        self, frame_path: str, frame_width: int, frame_height: int
    ) -> list[DetectionResult]:
        results = self.model(
            frame_path,
            device=self.device,
            conf=self.confidence,
            verbose=False,
        )
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                label = result.names[cls_id]
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bbox_x = int(x1)
                bbox_y = int(y1)
                bbox_w = int(x2 - x1)
                bbox_h = int(y2 - y1)

                detections.append(DetectionResult(
                    detector=self.name,
                    label=label,
                    confidence=conf,
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                    pixel_area=bbox_w * bbox_h,
                ))
        return detections

    def detect_batch(
        self, frame_paths: list[str], frame_width: int, frame_height: int
    ) -> list[list[DetectionResult]]:
        results = self.model(
            frame_paths,
            device=self.device,
            conf=self.confidence,
            verbose=False,
        )
        all_detections = []
        for result in results:
            frame_dets = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    label = result.names[cls_id]
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    bbox_x = int(x1)
                    bbox_y = int(y1)
                    bbox_w = int(x2 - x1)
                    bbox_h = int(y2 - y1)
                    frame_dets.append(DetectionResult(
                        detector=self.name,
                        label=label,
                        confidence=conf,
                        bbox_x=bbox_x,
                        bbox_y=bbox_y,
                        bbox_w=bbox_w,
                        bbox_h=bbox_h,
                        pixel_area=bbox_w * bbox_h,
                    ))
            all_detections.append(frame_dets)
        return all_detections
