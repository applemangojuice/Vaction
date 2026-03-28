# Vaction

Local video object categorization, indexing, and search over TV episodes.

Vaction processes TV episode files on your Mac, extracts visual content from sampled frames, categorizes what appears on screen, stores those categorizations in a searchable index, and returns episode-level metrics based on **pixel-time share**.

## Features

- **Ingest** local video files organized by series/season/episode
- **Sample frames** at configurable rates (1-10 fps)
- **Detect objects** using YOLOv8 (people, cars, desks, etc.)
- **Classify scenes** using CLIP (office, hallway, exterior, etc.)
- **Recognize characters** using face embeddings (InsightFace)
- **Search** with boolean queries: `"mark AND hallway"`, `"vehicle NOT office"`
- **Pixel-time metrics**: percentage of episode represented by each search term

## Requirements

- Python 3.10+
- FFmpeg (`brew install ffmpeg`)
- macOS with Apple Silicon recommended (MPS acceleration)

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# 1. Ingest a directory of episodes
vaction ingest /path/to/Severance/Season\ 1 --series Severance --fps 2

# 2. Run object and scene detection
vaction detect --series Severance --detectors yolo,clip

# 3. Enroll character faces (optional)
vaction enroll Severance "Mark" ref_images/mark1.jpg ref_images/mark2.jpg

# 4. Re-run detection with face recognition
vaction detect --series Severance --detectors face

# 5. Create search aliases
vaction alias vehicle car truck bus van
vaction alias office desk chair monitor cubicle

# 6. Search
vaction search "person" -s Severance
vaction search "mark AND hallway" -s Severance --verbose
vaction search "vehicle OR elevator" -s Severance --format json

# 7. Season report
vaction report "hallway" -s Severance
```

## Output Example

```
Query: "hallway"

  S01E01 - Pilot
    Episode share: 6.4%
    Strongest windows: 00:08:20-00:10:05, 00:34:10-00:36:40

  S01E02 - Half Loop
    Episode share: 8.9%
    Strongest windows: 00:12:00-00:14:30

  S01E03 - In Perpetuity
    Episode share: 7.1%
    Strongest windows: 00:05:45-00:08:10
```

## Pixel-Time Metric

For a query term `q`:

- **Pixel-time** = Σ (pixel_area × Δt) across all matching frames
- **Episode share** = Σ matched_pixels / (num_frames × width × height)

This produces interpretable percentages like "4.9% of the episode" rather than raw pixel counts.

## Architecture

```
vaction/
├── cli.py          # Click CLI commands
├── config.py       # Configuration management
├── db.py           # SQLite schema and CRUD
├── models.py       # Data models
├── ingest.py       # Video scanning and registration
├── sampler.py      # Frame extraction via ffmpeg
├── search.py       # Boolean query parser and SQL builder
├── metrics.py      # Pixel-time computation
├── export.py       # Output formatting (text/JSON/CSV)
└── detectors/
    ├── registry.py # Detector orchestration
    ├── yolo.py     # YOLOv8 object detection
    ├── clip.py     # CLIP scene classification
    └── faces.py    # Face detection and recognition
```

## Database

SQLite database stored at `~/.vaction/vaction.db` with tables:
- `series` → `season` → `episode` → `frame` → `detection`
- `character` → `character_embedding` (for face recognition)
- `alias` (search term expansion)
