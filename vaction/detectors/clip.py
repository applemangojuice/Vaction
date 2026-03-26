"""CLIP-based scene and attribute classification."""

from __future__ import annotations

from vaction.models import DetectionResult

# Default scene/attribute prompts for classification
DEFAULT_SCENE_PROMPTS = [
    "an office room",
    "a hallway or corridor",
    "a stairwell",
    "a car interior",
    "an exterior road or street",
    "a conference room",
    "a cubicle workspace",
    "a security desk or checkpoint",
    "an elevator",
    "a kitchen or break room",
    "a lobby or reception area",
    "a parking lot",
    "a bathroom or restroom",
    "a bedroom",
    "a living room",
]

DEFAULT_ATTRIBUTE_PROMPTS = [
    "a close-up shot",
    "a wide shot",
    "a dark or dimly lit scene",
    "a bright or well-lit scene",
    "a crowded scene with many people",
    "an empty room or space",
    "a scene with red objects",
    "a scene with blue objects",
]

# Map prompt text to short labels
PROMPT_LABELS = {
    "an office room": "office",
    "a hallway or corridor": "hallway",
    "a stairwell": "stairwell",
    "a car interior": "car_interior",
    "an exterior road or street": "exterior_road",
    "a conference room": "conference_room",
    "a cubicle workspace": "cubicle",
    "a security desk or checkpoint": "security_desk",
    "an elevator": "elevator",
    "a kitchen or break room": "kitchen",
    "a lobby or reception area": "lobby",
    "a parking lot": "parking_lot",
    "a bathroom or restroom": "bathroom",
    "a bedroom": "bedroom",
    "a living room": "living_room",
    "a close-up shot": "close_up",
    "a wide shot": "wide_shot",
    "a dark or dimly lit scene": "dark",
    "a bright or well-lit scene": "bright",
    "a crowded scene with many people": "crowded",
    "an empty room or space": "empty",
    "a scene with red objects": "red",
    "a scene with blue objects": "blue",
}


class CLIPDetector:
    """Scene and attribute classification using CLIP."""

    name = "clip"

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
        threshold: float = 0.2,
        scene_prompts: list[str] | None = None,
        attribute_prompts: list[str] | None = None,
        custom_prompts: list[dict] | None = None,
    ):
        import open_clip
        import torch

        self.device = device
        self.threshold = threshold

        # If custom_prompts provided (from DB), use those instead of defaults
        if custom_prompts:
            self.prompts = [p["prompt"] for p in custom_prompts]
            # Build label map from custom prompts
            for p in custom_prompts:
                if p["prompt"] not in PROMPT_LABELS:
                    PROMPT_LABELS[p["prompt"]] = p["label"]
        else:
            self.prompts = (scene_prompts or DEFAULT_SCENE_PROMPTS) + (
                attribute_prompts or DEFAULT_ATTRIBUTE_PROMPTS
            )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Pre-compute text embeddings
        with torch.no_grad():
            tokens = self.tokenizer(self.prompts).to(device)
            self.text_features = self.model.encode_text(tokens)
            self.text_features = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )

    def detect(
        self, frame_path: str, frame_width: int, frame_height: int
    ) -> list[DetectionResult]:
        import torch
        from PIL import Image

        image = Image.open(frame_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ self.text_features.T).squeeze(0)

        detections = []
        for i, (prompt, sim) in enumerate(zip(self.prompts, similarities)):
            score = float(sim.item())
            if score >= self.threshold:
                label = PROMPT_LABELS.get(prompt, prompt.lower().replace(" ", "_"))
                detections.append(DetectionResult(
                    detector=self.name,
                    label=label,
                    confidence=score,
                    bbox_x=None,
                    bbox_y=None,
                    bbox_w=None,
                    bbox_h=None,
                    pixel_area=frame_width * frame_height,  # Scene-level: full frame
                    attributes={"prompt": prompt},
                ))
        return detections

    def detect_batch(
        self, frame_paths: list[str], frame_width: int, frame_height: int
    ) -> list[list[DetectionResult]]:
        import torch
        from PIL import Image

        images = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            images.append(self.preprocess(img))

        batch = torch.stack(images).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ self.text_features.T

        all_detections = []
        for frame_idx in range(len(frame_paths)):
            frame_dets = []
            for i, prompt in enumerate(self.prompts):
                score = float(similarities[frame_idx, i].item())
                if score >= self.threshold:
                    label = PROMPT_LABELS.get(prompt, prompt.lower().replace(" ", "_"))
                    frame_dets.append(DetectionResult(
                        detector=self.name,
                        label=label,
                        confidence=score,
                        bbox_x=None,
                        bbox_y=None,
                        bbox_w=None,
                        bbox_h=None,
                        pixel_area=frame_width * frame_height,
                        attributes={"prompt": prompt},
                    ))
            all_detections.append(frame_dets)
        return all_detections
