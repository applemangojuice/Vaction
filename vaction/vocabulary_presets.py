"""Comprehensive detection vocabulary with low/medium/high presets."""

# Each entry: (prompt, label, category, preset_level)
# preset_level: 1=low (essentials), 2=medium, 3=high (everything)

VOCABULARY = [
    # ── PEOPLE & BODIES ─────────────────────────────────────────────────────
    ("a person or human figure", "person", "people", 1),
    ("a group of people", "group_of_people", "people", 2),
    ("a crowd of people", "crowd", "people", 3),
    ("a person's face in close-up", "face_closeup", "people", 1),
    ("a person from behind", "person_from_behind", "people", 3),
    ("a person's hands", "hands", "people", 3),
    ("a silhouette of a person", "silhouette", "people", 3),
    ("a child or young person", "child", "people", 2),

    # ── ACTIONS & BODY LANGUAGE ─────────────────────────────────────────────
    ("a person walking", "walking", "action", 2),
    ("a person running", "running", "action", 2),
    ("a person sitting", "sitting", "action", 2),
    ("a person standing", "standing", "action", 3),
    ("a person talking or speaking", "talking", "action", 1),
    ("a person crying or in distress", "crying", "action", 2),
    ("a person laughing or smiling", "laughing", "action", 2),
    ("people fighting or in physical conflict", "fighting", "action", 2),
    ("people hugging or embracing", "hugging", "action", 2),
    ("a person eating or drinking", "eating", "action", 2),
    ("a person typing or using a computer", "typing", "action", 3),
    ("a person reading", "reading", "action", 3),
    ("a person looking through a window", "looking_through_window", "action", 3),
    ("a person sleeping or lying down", "sleeping", "action", 3),
    ("a person pointing", "pointing", "action", 3),
    ("a person dancing", "dancing", "action", 3),
    ("a person whispering", "whispering", "action", 3),
    ("a person on the phone", "on_phone", "action", 2),
    ("a person driving", "driving", "action", 2),

    # ── VEHICLES ────────────────────────────────────────────────────────────
    ("a car or automobile", "car", "vehicle", 1),
    ("a truck or large vehicle", "truck", "vehicle", 2),
    ("a bus", "bus", "vehicle", 3),
    ("a van", "van", "vehicle", 3),
    ("a motorcycle or motorbike", "motorcycle", "vehicle", 3),
    ("a bicycle", "bicycle", "vehicle", 3),
    ("a taxi or cab", "taxi", "vehicle", 3),
    ("a police car", "police_car", "vehicle", 2),
    ("an ambulance", "ambulance", "vehicle", 3),
    ("a helicopter", "helicopter", "vehicle", 3),
    ("a boat or ship", "boat", "vehicle", 3),
    ("an airplane or aircraft", "airplane", "vehicle", 3),
    ("a train or subway", "train", "vehicle", 3),
    ("a SUV or sports utility vehicle", "suv", "vehicle", 3),
    ("an emergency vehicle with lights", "emergency_vehicle", "vehicle", 3),

    # ── INDOOR SPACES ───────────────────────────────────────────────────────
    ("an office room or workspace", "office", "scene", 1),
    ("a hallway or corridor", "hallway", "scene", 1),
    ("a stairwell or staircase", "stairwell", "scene", 1),
    ("an elevator or lift", "elevator", "scene", 1),
    ("a conference room or meeting room", "conference_room", "scene", 2),
    ("a cubicle or open-plan workspace", "cubicle", "scene", 2),
    ("a security desk or checkpoint", "security_desk", "scene", 2),
    ("a kitchen or break room", "kitchen", "scene", 2),
    ("a lobby or reception area", "lobby", "scene", 2),
    ("a bathroom or restroom", "bathroom", "scene", 2),
    ("a bedroom", "bedroom", "scene", 2),
    ("a living room or lounge", "living_room", "scene", 2),
    ("a dining room", "dining_room", "scene", 3),
    ("a hospital or medical room", "hospital", "scene", 2),
    ("a restaurant or bar", "restaurant", "scene", 2),
    ("a classroom or lecture hall", "classroom", "scene", 3),
    ("a library", "library", "scene", 3),
    ("a gym or fitness center", "gym", "scene", 3),
    ("a store or shop", "store", "scene", 3),
    ("a warehouse or storage room", "warehouse", "scene", 3),
    ("a basement", "basement", "scene", 3),
    ("a closet or small room", "closet", "scene", 3),
    ("a laundry room", "laundry_room", "scene", 3),
    ("a garage", "garage", "scene", 3),
    ("a prison cell or jail", "prison", "scene", 3),
    ("a server room or data center", "server_room", "scene", 3),

    # ── OUTDOOR SPACES ──────────────────────────────────────────────────────
    ("an exterior road or street", "exterior_road", "scene", 1),
    ("a parking lot or parking garage", "parking_lot", "scene", 2),
    ("a garden or yard", "garden", "scene", 2),
    ("a forest or woods", "forest", "scene", 2),
    ("a beach or waterfront", "beach", "scene", 3),
    ("an urban city street", "city_street", "scene", 2),
    ("a suburban neighborhood", "suburban", "scene", 3),
    ("a rural or countryside scene", "rural", "scene", 3),
    ("a rooftop", "rooftop", "scene", 3),
    ("a bridge", "bridge", "scene", 3),
    ("a park or playground", "park", "scene", 2),
    ("a construction site", "construction_site", "scene", 3),
    ("a gas station", "gas_station", "scene", 3),

    # ── FURNITURE & OBJECTS ─────────────────────────────────────────────────
    ("a desk or table", "desk", "object", 1),
    ("a chair or seat", "chair", "object", 2),
    ("a couch or sofa", "couch", "object", 2),
    ("a bed", "bed", "object", 2),
    ("a door", "door", "object", 1),
    ("a window", "window", "object", 2),
    ("stairs or steps", "stairs", "object", 2),

    # ── TECHNOLOGY ──────────────────────────────────────────────────────────
    ("a computer screen or monitor", "screen", "object", 1),
    ("a laptop computer", "laptop", "object", 2),
    ("a phone or smartphone", "phone", "object", 1),
    ("a television or TV screen", "television", "object", 2),
    ("a camera or surveillance camera", "camera", "object", 3),
    ("a printer or office machine", "printer", "object", 3),

    # ── DOCUMENTS & SIGNS ──────────────────────────────────────────────────
    ("a badge or ID card", "badge", "object", 2),
    ("a book or document", "book", "object", 2),
    ("signage or signs", "signage", "object", 2),
    ("a letter or envelope", "letter", "object", 3),
    ("a newspaper", "newspaper", "object", 3),
    ("a clipboard", "clipboard", "object", 3),

    # ── FOOD & DRINK ────────────────────────────────────────────────────────
    ("food or a meal", "food", "object", 2),
    ("a drink or beverage", "drink", "object", 2),
    ("a coffee cup", "coffee", "object", 3),
    ("a bottle", "bottle", "object", 3),

    # ── WEAPONS & DANGER ────────────────────────────────────────────────────
    ("a weapon or firearm", "weapon", "object", 2),
    ("a knife or blade", "knife", "object", 3),
    ("blood or injury", "blood", "attribute", 2),
    ("fire or flames", "fire", "object", 3),
    ("an explosion", "explosion", "object", 3),

    # ── CLOTHING & ACCESSORIES ──────────────────────────────────────────────
    ("a suit or formal clothing", "suit", "object", 3),
    ("a uniform", "uniform", "object", 3),
    ("glasses or eyewear", "glasses", "object", 3),
    ("a hat or cap", "hat", "object", 3),
    ("a mask", "mask", "object", 3),

    # ── ANIMALS ─────────────────────────────────────────────────────────────
    ("a dog", "dog", "object", 3),
    ("a cat", "cat", "object", 3),

    # ── SHOT TYPES / CINEMATOGRAPHY ─────────────────────────────────────────
    ("a close-up shot of a face", "close_up", "shot_type", 1),
    ("a wide establishing shot", "wide_shot", "shot_type", 1),
    ("a medium shot of a person", "medium_shot", "shot_type", 2),
    ("an overhead or bird's-eye view", "overhead_shot", "shot_type", 3),
    ("a point-of-view shot", "pov_shot", "shot_type", 3),
    ("a two-shot of two people", "two_shot", "shot_type", 2),
    ("a group shot with multiple people", "group_shot", "shot_type", 3),
    ("over-the-shoulder shot", "over_shoulder", "shot_type", 3),

    # ── LIGHTING & MOOD ────────────────────────────────────────────────────
    ("a dark or dimly lit scene", "dark", "attribute", 1),
    ("a bright or well-lit scene", "bright", "attribute", 1),
    ("a scene with dramatic shadows", "dramatic_shadows", "attribute", 3),
    ("a scene with natural sunlight", "natural_light", "attribute", 3),
    ("a fluorescent-lit interior", "fluorescent", "attribute", 3),
    ("a nighttime scene", "nighttime", "attribute", 2),
    ("a daytime scene", "daytime", "attribute", 2),

    # ── COLORS ──────────────────────────────────────────────────────────────
    ("a scene with predominantly red colors", "red", "color", 2),
    ("a scene with predominantly blue colors", "blue", "color", 2),
    ("a scene with predominantly green colors", "green", "color", 3),
    ("a scene with predominantly white or sterile aesthetic", "white_sterile", "color", 2),
    ("a scene with predominantly dark or black aesthetic", "dark_aesthetic", "color", 3),
    ("a warm-toned scene with orange or yellow", "warm_tones", "color", 3),
    ("a cool-toned scene with blue or grey", "cool_tones", "color", 3),

    # ── ATMOSPHERE ──────────────────────────────────────────────────────────
    ("a crowded scene with many people", "crowded", "attribute", 1),
    ("an empty room or space", "empty", "attribute", 1),
    ("a tense or suspenseful scene", "tense", "attribute", 2),
    ("a calm or peaceful scene", "calm", "attribute", 2),
    ("a chaotic or disordered scene", "chaotic", "attribute", 3),
    ("a clean and organized space", "organized", "attribute", 3),
    ("a messy or cluttered space", "cluttered", "attribute", 3),
    ("a scene with rain or water", "rain", "attribute", 3),
    ("a scene with snow or winter", "snow", "attribute", 3),
    ("a foggy or misty scene", "fog", "attribute", 3),
    ("a scene with mirrors or reflections", "mirrors", "attribute", 3),
]

# Default themes
DEFAULT_THEMES = [
    {
        "name": "vehicular_action",
        "description": "Vehicle-related action and transportation",
        "labels": ["car", "motorcycle", "truck", "bus", "van", "taxi", "police_car", "bicycle", "boat", "helicopter", "airplane", "train", "suv", "emergency_vehicle", "driving"],
    },
    {
        "name": "combat",
        "description": "Fighting, weapons, and violent action",
        "labels": ["fighting", "weapon", "knife", "explosion", "fire", "running", "blood"],
    },
    {
        "name": "human_presence",
        "description": "People and their visual framing",
        "labels": ["person", "face_closeup", "close_up", "group_of_people", "crowd", "two_shot", "group_shot"],
    },
    {
        "name": "social_interaction",
        "description": "Interpersonal and social activities",
        "labels": ["talking", "sitting", "hugging", "laughing", "on_phone", "reading", "eating", "group_of_people", "two_shot"],
    },
    {
        "name": "technology",
        "description": "Screens, devices, and tech objects",
        "labels": ["laptop", "phone", "printer", "screen", "television", "camera", "typing"],
    },
    {
        "name": "outdoor_environment",
        "description": "Outdoor locations and wide environments",
        "labels": ["beach", "forest", "exterior_road", "city_street", "rooftop", "garden", "park", "rural", "suburban", "parking_lot", "wide_shot"],
    },
    {
        "name": "interior_domestic",
        "description": "Indoor rooms and built spaces",
        "labels": ["office", "hallway", "kitchen", "bedroom", "living_room", "dining_room", "lobby", "bathroom", "conference_room", "cubicle", "warehouse", "basement"],
    },
    {
        "name": "emotion",
        "description": "Emotional expression and mood",
        "labels": ["crying", "hugging", "laughing", "tense", "calm"],
    },
    {
        "name": "pursuit_motion",
        "description": "Chase sequences and movement",
        "labels": ["running", "driving", "walking", "car", "motorcycle", "helicopter", "airplane", "chaotic"],
    },
    {
        "name": "spectacle_scale",
        "description": "Large-scale dramatic visuals",
        "labels": ["explosion", "crowd", "airplane", "helicopter", "wide_shot", "fire", "chaotic"],
    },
]


def get_prompts_for_level(level: int) -> list[dict]:
    """Get all prompts at or below the given level (1=low, 2=medium, 3=high)."""
    return [
        {"prompt": prompt, "label": label, "category": category}
        for prompt, label, category, plevel in VOCABULARY
        if plevel <= level
    ]


def get_level_counts() -> dict:
    """Return counts for each preset level."""
    low = sum(1 for _, _, _, l in VOCABULARY if l <= 1)
    med = sum(1 for _, _, _, l in VOCABULARY if l <= 2)
    high = sum(1 for _, _, _, l in VOCABULARY if l <= 3)
    return {"low": low, "medium": med, "high": high}
