"""Output formatting for search results."""

from __future__ import annotations

import csv
import io
import json
from typing import TextIO

from vaction.metrics import format_timestamp
from vaction.models import SearchResult


def format_results_text(results: list[SearchResult], verbose: bool = False) -> str:
    """Format search results as human-readable text."""
    if not results:
        return "No results found."

    lines = []
    query = results[0].query if results else ""
    lines.append(f'Query: "{query}"')
    lines.append("")

    for r in results:
        ep_label = f"S{r.season_number:02d}E{r.episode_number:02d}"
        if r.episode_title:
            ep_label += f" - {r.episode_title}"

        lines.append(f"  {ep_label}")
        lines.append(f"    Episode share: {r.episode_share_pct:.1f}%")

        if verbose:
            lines.append(f"    Pixel-time: {r.pixel_time:,.0f} pixel-seconds")
            lines.append(f"    Pixel-minutes: {r.pixel_minutes:,.1f}")

        if r.strongest_windows:
            windows_str = ", ".join(
                f"{format_timestamp(start)}-{format_timestamp(end)}"
                for start, end in r.strongest_windows
            )
            lines.append(f"    Strongest windows: {windows_str}")

        lines.append("")

    return "\n".join(lines)


def format_results_json(results: list[SearchResult]) -> str:
    """Format search results as JSON."""
    output = []
    for r in results:
        entry = {
            "series": r.series_name,
            "season": r.season_number,
            "episode": r.episode_number,
            "title": r.episode_title,
            "query": r.query,
            "episode_share_pct": round(r.episode_share_pct, 2),
            "pixel_time_seconds": round(r.pixel_time, 2),
            "pixel_minutes": round(r.pixel_minutes, 2),
            "strongest_windows": [
                {
                    "start": format_timestamp(start),
                    "end": format_timestamp(end),
                    "start_secs": round(start, 2),
                    "end_secs": round(end, 2),
                }
                for start, end in r.strongest_windows
            ],
        }
        output.append(entry)
    return json.dumps(output, indent=2)


def format_results_csv(results: list[SearchResult]) -> str:
    """Format search results as CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "series", "season", "episode", "title", "query",
        "episode_share_pct", "pixel_time_seconds", "pixel_minutes",
        "strongest_window_1", "strongest_window_2", "strongest_window_3",
    ])

    for r in results:
        windows = []
        for start, end in r.strongest_windows[:3]:
            windows.append(f"{format_timestamp(start)}-{format_timestamp(end)}")
        while len(windows) < 3:
            windows.append("")

        writer.writerow([
            r.series_name,
            r.season_number,
            r.episode_number,
            r.episode_title,
            r.query,
            round(r.episode_share_pct, 2),
            round(r.pixel_time, 2),
            round(r.pixel_minutes, 2),
            *windows,
        ])

    return buf.getvalue()
