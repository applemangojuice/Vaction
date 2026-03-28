"""Command-line interface for Vaction."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from vaction import __version__
from vaction.config import VactionConfig, get_config
from vaction.db import db_session, init_db, insert_alias


@click.group()
@click.version_option(version=__version__)
@click.option("--data-dir", type=click.Path(), default=None, help="Data directory (default: ~/.vaction)")
@click.option("--db", "db_name", default=None, help="Database filename")
@click.pass_context
def cli(ctx, data_dir, db_name):
    """Vaction: Local video object categorization, indexing, and search."""
    overrides = {}
    if data_dir:
        overrides["data_dir"] = Path(data_dir)
    if db_name:
        overrides["db_name"] = db_name

    config = get_config(**overrides)
    config.ensure_dirs()
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--series", "-s", required=True, help="Series name")
@click.option("--fps", type=float, default=None, help="Frame sampling rate (default: 2.0)")
@click.option("--transcode", type=int, default=None, help="Transcode to this height (e.g., 720)")
@click.option("--no-sample", is_flag=True, help="Skip frame sampling (register only)")
@click.pass_context
def ingest(ctx, path, series, fps, transcode, no_sample):
    """Ingest video files from a directory.

    Scans PATH for video files, registers them in the database,
    optionally transcodes, and samples frames.
    """
    from tqdm import tqdm

    from vaction.ingest import ingest_directory
    from vaction.sampler import sample_episodes

    config: VactionConfig = ctx.obj["config"]
    if fps:
        config = get_config(default_fps=fps, data_dir=config.data_dir)
    if transcode:
        config = get_config(
            transcode_resolution=transcode,
            default_fps=fps or config.default_fps,
            data_dir=config.data_dir,
        )

    root = Path(path).resolve()
    click.echo(f"Scanning {root} for video files...")

    with db_session(config.db_path) as conn:
        episode_ids = ingest_directory(conn, root, series, config)
        click.echo(f"Registered {len(episode_ids)} episodes.")

        if not no_sample and episode_ids:
            click.echo(f"Sampling frames at {config.default_fps} fps...")
            with tqdm(total=len(episode_ids), desc="Sampling") as pbar:
                def progress(ep_id, count):
                    pbar.update(1)
                    pbar.set_postfix(frames=count)

                results = sample_episodes(
                    conn, episode_ids, config.frames_dir,
                    fps=config.default_fps, progress_callback=progress,
                )

            total_frames = sum(results.values())
            click.echo(f"Extracted {total_frames} frames from {len(results)} episodes.")


@cli.command()
@click.option("--series", "-s", required=True, help="Series name")
@click.option("--episode", "-e", default=None, help="Specific episode (e.g., S01E03)")
@click.option("--detectors", "-d", default="yolo,clip", help="Comma-separated detectors to run")
@click.option("--batch-size", type=int, default=16, help="Batch size for inference")
@click.option("--device", default=None, help="Device: cpu, mps, cuda")
@click.option("--confidence", type=float, default=0.25, help="Minimum confidence threshold")
@click.pass_context
def detect(ctx, series, episode, detectors, batch_size, device, confidence):
    """Run detection on sampled frames.

    Analyzes frames using the specified detectors (yolo, clip, face)
    and stores detections in the index.
    """
    from tqdm import tqdm

    from vaction.detectors.registry import DetectorRegistry

    config: VactionConfig = ctx.obj["config"]
    dev = device or config.device

    registry = DetectorRegistry()
    detector_names = [d.strip() for d in detectors.split(",")]

    for name in detector_names:
        if name == "yolo":
            from vaction.detectors.yolo import YOLODetector
            registry.register(YOLODetector(
                model_name=config.yolo_model, device=dev, confidence=confidence
            ))
        elif name == "clip":
            from vaction.detectors.clip import CLIPDetector
            registry.register(CLIPDetector(
                model_name=config.clip_model,
                pretrained=config.clip_pretrained,
                device=dev,
            ))
        elif name == "face":
            from vaction.detectors.faces import FaceDetector
            det = FaceDetector(device=dev, distance_threshold=config.face_distance_threshold)
            registry.register(det)
        else:
            click.echo(f"Unknown detector: {name}", err=True)
            sys.exit(1)

    click.echo(f"Running detectors: {', '.join(registry.detector_names)}")

    with db_session(config.db_path) as conn:
        # Load character embeddings for face detector
        if "face" in detector_names:
            row = conn.execute("SELECT id FROM series WHERE name = ?", (series,)).fetchone()
            if row:
                for det in registry._detectors:
                    if hasattr(det, "load_character_embeddings"):
                        det.load_character_embeddings(conn, row["id"])

        # Get frames to process
        query = """
            SELECT f.id, f.file_path, e.width, e.height
            FROM frame f
            JOIN episode e ON f.episode_id = e.id
            JOIN season s ON e.season_id = s.id
            JOIN series sr ON s.series_id = sr.id
            WHERE sr.name = ?
              AND f.file_path IS NOT NULL
        """
        params = [series]

        if episode:
            import re
            m = re.match(r"[Ss](\d+)[Ee](\d+)", episode)
            if m:
                query += " AND s.number = ? AND e.number = ?"
                params.extend([int(m.group(1)), int(m.group(2))])

        # Exclude already-detected frames
        query += " AND f.id NOT IN (SELECT DISTINCT frame_id FROM detection)"
        query += " ORDER BY f.id"

        frames = conn.execute(query, params).fetchall()
        if not frames:
            click.echo("No unprocessed frames found.")
            return

        click.echo(f"Processing {len(frames)} frames...")
        total_detections = 0

        # Process in batches
        for i in tqdm(range(0, len(frames), batch_size), desc="Detecting"):
            batch = frames[i:i + batch_size]
            batch_dicts = [
                {"id": f["id"], "file_path": f["file_path"]}
                for f in batch
            ]
            width = batch[0]["width"]
            height = batch[0]["height"]

            count = registry.run_on_batch(conn, batch_dicts, width, height)
            total_detections += count
            conn.commit()

        click.echo(f"Created {total_detections} detections.")


@cli.command()
@click.argument("series")
@click.argument("character_name")
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--device", default=None, help="Device: cpu, mps, cuda")
@click.pass_context
def enroll(ctx, series, character_name, images, device):
    """Enroll a character for face recognition.

    Provide one or more reference images of the character's face.
    """
    from vaction.detectors.faces import enroll_character

    config: VactionConfig = ctx.obj["config"]
    dev = device or config.device

    with db_session(config.db_path) as conn:
        row = conn.execute("SELECT id FROM series WHERE name = ?", (series,)).fetchone()
        if not row:
            click.echo(f"Series '{series}' not found. Ingest videos first.", err=True)
            sys.exit(1)

        count = enroll_character(
            conn, row["id"], character_name, list(images), device=dev
        )
        click.echo(f"Enrolled {count} face embedding(s) for '{character_name}'.")


@cli.command()
@click.argument("query")
@click.option("--series", "-s", required=True, help="Series name")
@click.option("--season", type=int, default=None, help="Filter by season number")
@click.option("--episode", "-e", type=int, default=None, help="Filter by episode number")
@click.option("--confidence", type=float, default=0.25, help="Minimum confidence threshold")
@click.option("--format", "output_format", type=click.Choice(["text", "json", "csv"]), default="text")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed metrics")
@click.pass_context
def search(ctx, query, series, season, episode, confidence, output_format, verbose):
    """Search the visual index for a term or boolean expression.

    Examples:
        vaction search "car" -s Severance
        vaction search "mark AND hallway" -s Severance
        vaction search "helly OR irving" -s Severance --format json
    """
    from vaction.export import format_results_csv, format_results_json, format_results_text
    from vaction.metrics import compute_episode_metrics

    config: VactionConfig = ctx.obj["config"]

    with db_session(config.db_path) as conn:
        row = conn.execute("SELECT id FROM series WHERE name = ?", (series,)).fetchone()
        if not row:
            click.echo(f"Series '{series}' not found.", err=True)
            sys.exit(1)

        results = compute_episode_metrics(
            conn, query, confidence,
            series_id=row["id"],
            season_number=season,
            episode_number=episode,
        )

        if output_format == "json":
            click.echo(format_results_json(results))
        elif output_format == "csv":
            click.echo(format_results_csv(results))
        else:
            click.echo(format_results_text(results, verbose=verbose))


@cli.command()
@click.argument("query")
@click.option("--series", "-s", required=True, help="Series name")
@click.option("--confidence", type=float, default=0.25, help="Minimum confidence threshold")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def report(ctx, query, series, confidence, output_format):
    """Generate a season-level summary report for a query."""
    import json as json_mod

    from vaction.export import format_results_text
    from vaction.metrics import compute_season_summary

    config: VactionConfig = ctx.obj["config"]

    with db_session(config.db_path) as conn:
        row = conn.execute("SELECT id FROM series WHERE name = ?", (series,)).fetchone()
        if not row:
            click.echo(f"Series '{series}' not found.", err=True)
            sys.exit(1)

        summary = compute_season_summary(conn, query, row["id"], confidence)

        if output_format == "json":
            # Convert SearchResult objects for JSON serialization
            output = {
                "query": summary["query"],
                "total_pixel_time_seconds": round(summary["total_pixel_time"], 2),
                "total_pixel_minutes": round(summary.get("total_pixel_minutes", 0), 2),
                "average_episode_share_pct": summary["average_episode_share_pct"],
                "num_matching_episodes": summary.get("num_matching_episodes", 0),
            }
            click.echo(json_mod.dumps(output, indent=2))
        else:
            click.echo(f"Season Summary for: \"{query}\"")
            click.echo(f"  Average episode share: {summary['average_episode_share_pct']:.1f}%")
            click.echo(f"  Matching episodes: {summary.get('num_matching_episodes', 0)}")
            click.echo()
            if summary["episodes"]:
                click.echo(format_results_text(summary["episodes"]))


@cli.command()
@click.argument("term")
@click.argument("labels", nargs=-1, required=True)
@click.pass_context
def alias(ctx, term, labels):
    """Create a search alias.

    Maps a search term to one or more detection labels.

    Examples:
        vaction alias vehicle car truck bus van
        vaction alias office desk chair monitor cubicle
    """
    config: VactionConfig = ctx.obj["config"]

    with db_session(config.db_path) as conn:
        for label in labels:
            insert_alias(conn, term, label)
        click.echo(f"Alias '{term}' -> {', '.join(labels)}")


@cli.command()
@click.option("--series", "-s", default=None, help="Filter by series")
@click.pass_context
def status(ctx, series):
    """Show database status and statistics."""
    config: VactionConfig = ctx.obj["config"]

    with db_session(config.db_path) as conn:
        # Series count
        if series:
            rows = conn.execute(
                "SELECT * FROM series WHERE name = ?", (series,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM series").fetchall()

        if not rows:
            click.echo("No series found. Use 'vaction ingest' to add videos.")
            return

        for sr in rows:
            click.echo(f"\nSeries: {sr['name']}")
            click.echo(f"  Path: {sr['root_path']}")

            seasons = conn.execute(
                "SELECT * FROM season WHERE series_id = ? ORDER BY number",
                (sr["id"],),
            ).fetchall()

            total_episodes = 0
            total_frames = 0
            total_detections = 0

            for s in seasons:
                eps = conn.execute(
                    "SELECT * FROM episode WHERE season_id = ? ORDER BY number",
                    (s["id"],),
                ).fetchall()

                click.echo(f"  Season {s['number']}: {len(eps)} episodes")

                for ep in eps:
                    total_episodes += 1
                    total_frames += ep["num_frames"]

                    det_count = conn.execute(
                        """SELECT COUNT(*) as c FROM detection d
                        JOIN frame f ON d.frame_id = f.id
                        WHERE f.episode_id = ?""",
                        (ep["id"],),
                    ).fetchone()["c"]
                    total_detections += det_count

            click.echo(f"  Total: {total_episodes} episodes, {total_frames} frames, {total_detections} detections")

        # Show aliases
        aliases = conn.execute("SELECT term, GROUP_CONCAT(label, ', ') as labels FROM alias GROUP BY term").fetchall()
        if aliases:
            click.echo("\nAliases:")
            for a in aliases:
                click.echo(f"  {a['term']} -> {a['labels']}")


if __name__ == "__main__":
    cli()
