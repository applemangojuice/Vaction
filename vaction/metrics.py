"""Pixel-time metrics computation and aggregation."""

from __future__ import annotations

from collections import defaultdict

from vaction.models import SearchResult
from vaction.search import execute_search


def compute_episode_metrics(
    conn,
    query_str: str,
    confidence_threshold: float = 0.25,
    series_id: int | None = None,
    season_number: int | None = None,
    episode_number: int | None = None,
) -> list[SearchResult]:
    """Compute pixel-time metrics aggregated per episode.

    Returns a list of SearchResult, one per matching episode.
    """
    rows = execute_search(
        conn, query_str, confidence_threshold,
        series_id=series_id,
        season_number=season_number,
        episode_number=episode_number,
    )

    if not rows:
        return []

    # Group by episode
    episodes: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        episodes[row["episode_id"]].append(row)

    results = []
    for ep_id, frames in episodes.items():
        first = frames[0]
        width = first["width"]
        height = first["height"]
        num_frames = first["num_frames"]
        frame_pixels = width * height

        # Pixel-time: sum of (matched_pixel_area * delta_t) across frames
        pixel_time = sum(f["matched_pixel_area"] * f["delta_t"] for f in frames)

        # Total pixel budget for the episode
        total_budget = num_frames * frame_pixels

        # Episode share: fraction of total pixel budget
        episode_share = sum(f["matched_pixel_area"] for f in frames) / total_budget if total_budget > 0 else 0.0

        # Find strongest contiguous windows
        strongest = _find_strongest_windows(frames, top_n=3)

        results.append(SearchResult(
            series_name=first["series_name"],
            season_number=first["season_number"],
            episode_number=first["episode_number"],
            episode_title=first.get("episode_title", ""),
            query=query_str,
            pixel_time=pixel_time,
            pixel_minutes=pixel_time / 60.0,
            episode_share=episode_share,
            episode_share_pct=episode_share * 100.0,
            total_pixel_budget=float(total_budget),
            strongest_windows=strongest,
        ))

    # Sort by episode share descending
    results.sort(key=lambda r: r.episode_share_pct, reverse=True)
    return results


def compute_season_summary(
    conn,
    query_str: str,
    series_id: int,
    confidence_threshold: float = 0.25,
) -> dict:
    """Compute aggregated metrics across a full season or series."""
    results = compute_episode_metrics(
        conn, query_str, confidence_threshold, series_id=series_id
    )

    if not results:
        return {
            "query": query_str,
            "total_pixel_time": 0.0,
            "average_episode_share_pct": 0.0,
            "episodes": [],
        }

    total_pt = sum(r.pixel_time for r in results)
    avg_share = sum(r.episode_share_pct for r in results) / len(results)

    return {
        "query": query_str,
        "total_pixel_time": total_pt,
        "total_pixel_minutes": total_pt / 60.0,
        "average_episode_share_pct": round(avg_share, 2),
        "num_matching_episodes": len(results),
        "episodes": results,
    }


def _find_strongest_windows(
    frames: list[dict], top_n: int = 3, min_gap: float = 5.0
) -> list[tuple[float, float]]:
    """Find the strongest contiguous time windows of detection.

    Groups consecutive frames where detection is present into windows,
    then returns the top_n by total pixel area.
    """
    if not frames:
        return []

    # Sort by timestamp
    sorted_frames = sorted(frames, key=lambda f: f["timestamp"])

    # Group into contiguous windows
    windows = []
    current_start = sorted_frames[0]["timestamp"]
    current_end = sorted_frames[0]["timestamp"]
    current_area = sorted_frames[0]["matched_pixel_area"]
    current_delta = sorted_frames[0]["delta_t"]

    for frame in sorted_frames[1:]:
        if frame["timestamp"] - current_end <= current_delta * 2:
            # Continue current window
            current_end = frame["timestamp"]
            current_area += frame["matched_pixel_area"]
        else:
            # Start new window
            windows.append((current_start, current_end + current_delta, current_area))
            current_start = frame["timestamp"]
            current_end = frame["timestamp"]
            current_area = frame["matched_pixel_area"]
            current_delta = frame["delta_t"]

    windows.append((current_start, current_end + current_delta, current_area))

    # Sort by total area descending, take top_n
    windows.sort(key=lambda w: w[2], reverse=True)
    return [(w[0], w[1]) for w in windows[:top_n]]


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
