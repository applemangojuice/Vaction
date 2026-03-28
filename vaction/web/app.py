"""Vaction Web UI — Flask application."""

from __future__ import annotations

import json
import os
import queue
import re
import sqlite3
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

from vaction.config import VactionConfig, get_config
from vaction.db import (
    db_session, get_aliases, get_connection, init_db, insert_alias,
    create_theme, update_theme, delete_theme, get_all_themes, get_theme_labels,
    get_clip_prompts, add_clip_prompt, delete_clip_prompt, toggle_clip_prompt,
    seed_default_clip_prompts, load_vocabulary_preset, seed_default_themes,
    mark_frames_detected_batch, get_unprocessed_frames, get_all_series_frames,
    clear_detection_progress, insert_detections_batch,
)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

# Global state for background jobs
_jobs: dict[str, dict] = {}
_job_queues: dict[str, queue.Queue] = {}
_job_logs: dict[str, list] = {}  # Store recent log messages per job
_config: VactionConfig | None = None
# GPU lock: serialize GPU-heavy operations (face detection + clustering can OOM if concurrent)
_gpu_lock = threading.Lock()


def _job_update(job_id: str, **kwargs):
    """Update a job's state. Thread-safe because GIL protects dict updates."""
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)


def _job_put(job_id: str, event: str, data: str, log: bool = True):
    """Put an event on the job queue AND update _jobs progress state."""
    if job_id in _job_queues:
        _job_queues[job_id].put({"event": event, "data": data})
    # Store latest progress in _jobs for polling
    if job_id in _jobs:
        if event == "progress":
            try:
                _jobs[job_id]["progress"] = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                pass
        elif event == "status":
            _jobs[job_id]["last_status"] = data
        elif event == "complete":
            _jobs[job_id]["status"] = "done"
            try:
                _jobs[job_id]["result"] = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                _jobs[job_id]["result"] = data
        elif event == "error":
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = data
    # Store log messages
    if log and job_id in _job_logs:
        _job_logs[job_id].append({"event": event, "data": data, "time": time.time()})


def get_app_config() -> VactionConfig:
    global _config
    if _config is None:
        _config = get_config()
        _config.ensure_dirs()
    return _config


def get_db():
    config = get_app_config()
    return init_db(config.db_path)


# ── Pages ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/content")
def content_page():
    return render_template("content.html")


@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")


@app.route("/settings")
def settings_page():
    return render_template("settings.html")


# Legacy routes (redirect to new pages)
@app.route("/add-videos")
def add_videos_page():
    return render_template("content.html")


@app.route("/detect")
def detect_page():
    return render_template("content.html")


@app.route("/processing")
def processing_page():
    return render_template("content.html")


@app.route("/search")
def search_page():
    return render_template("analysis.html")


@app.route("/charts")
def charts_page():
    return render_template("analysis.html")


@app.route("/insights")
def insights_page():
    return render_template("analysis.html")


@app.route("/explorer")
def explorer_page():
    return render_template("analysis.html")


@app.route("/vocabulary")
def vocabulary_page():
    return render_template("settings.html")


@app.route("/themes")
def themes_page():
    return render_template("settings.html")


@app.route("/faces")
def faces_page():
    return render_template("settings.html")


@app.route("/status")
def status_page():
    return render_template("settings.html")


# ── API: Status ──────────────────────────────────────────────────────────────


@app.route("/api/status")
def api_status():
    conn = get_db()
    try:
        series_list = conn.execute("SELECT * FROM series ORDER BY name").fetchall()
        result = []
        for sr in series_list:
            seasons = conn.execute(
                "SELECT * FROM season WHERE series_id = ? ORDER BY number", (sr["id"],)
            ).fetchall()

            total_episodes = 0
            total_frames = 0
            total_detections = 0
            episodes_list = []

            for s in seasons:
                eps = conn.execute(
                    "SELECT * FROM episode WHERE season_id = ? ORDER BY number", (s["id"],)
                ).fetchall()
                for ep in eps:
                    total_episodes += 1
                    total_frames += ep["num_frames"]
                    det_count = conn.execute(
                        "SELECT COUNT(*) as c FROM detection d JOIN frame f ON d.frame_id = f.id WHERE f.episode_id = ?",
                        (ep["id"],),
                    ).fetchone()["c"]
                    total_detections += det_count
                    episodes_list.append({
                        "season": s["number"],
                        "episode": ep["number"],
                        "title": ep["title"] or "",
                        "frames": ep["num_frames"],
                        "detections": det_count,
                        "duration": ep["duration_secs"],
                        "resolution": f"{ep['width']}x{ep['height']}",
                    })

            result.append({
                "name": sr["name"],
                "id": sr["id"],
                "path": sr["root_path"],
                "total_episodes": total_episodes,
                "total_frames": total_frames,
                "total_detections": total_detections,
                "episodes": episodes_list,
            })

        return jsonify({"series": result})
    finally:
        conn.close()


# ── API: Ingest ──────────────────────────────────────────────────────────────


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    data = request.json
    folder_path = data.get("path", "").strip()
    series_name = data.get("series", "").strip()
    fps = float(data.get("fps", 2.0))
    transcode = data.get("transcode")  # None or int like 720

    if not folder_path or not series_name:
        return jsonify({"error": "path and series are required"}), 400

    root = Path(folder_path).expanduser().resolve()
    if not root.exists():
        return jsonify({"error": f"Path does not exist: {root}"}), 400

    job_id = f"ingest_{int(time.time())}"
    _job_queues[job_id] = queue.Queue()
    _job_logs[job_id] = []
    _jobs[job_id] = {"type": "ingest", "status": "running", "series": series_name, "progress": None, "last_status": "", "page": "/add-videos"}

    def run_ingest():
        try:
            config = get_app_config()
            config_overrides = {"default_fps": fps, "data_dir": config.data_dir}
            if transcode:
                config_overrides["transcode_resolution"] = int(transcode)
            cfg = get_config(**config_overrides)
            cfg.ensure_dirs()

            _job_put(job_id, "status", "Scanning for video files...")

            from vaction.ingest import ingest_directory
            conn = init_db(cfg.db_path)
            try:
                episode_ids = ingest_directory(conn, root, series_name, cfg)
                _job_put(job_id, "status", f"Found {len(episode_ids)} episodes. Extracting frames at {fps} fps...")

                if episode_ids:
                    from vaction.sampler import sample_episodes
                    total_eps = len(episode_ids)

                    def progress_cb(ep_id, count):
                        row = conn.execute("SELECT number FROM episode WHERE id = ?", (ep_id,)).fetchone()
                        ep_num = row["number"] if row else ep_id
                        _job_put(job_id, "progress", json.dumps({
                            "episode": ep_num, "frames": count,
                            "done": False,
                            "pct": 0,  # ingest doesn't have a total for %
                        }))

                    results = sample_episodes(conn, episode_ids, cfg.frames_dir, fps=fps, progress_callback=progress_cb)
                    total_frames = sum(results.values())
                    _job_put(job_id, "complete", json.dumps({
                        "episodes": len(episode_ids),
                        "total_frames": total_frames,
                    }))
                else:
                    _job_put(job_id, "complete", json.dumps({"episodes": 0, "total_frames": 0}))
            finally:
                conn.close()
        except Exception as e:
            _job_put(job_id, "error", str(e))
        finally:
            _jobs[job_id]["status"] = "done"

    threading.Thread(target=run_ingest, daemon=True).start()
    return jsonify({"job_id": job_id})


# ── API: Detection ───────────────────────────────────────────────────────────


@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.json
    series_name = data.get("series", "").strip()
    detectors = data.get("detectors", "yolo,clip")
    batch_size = int(data.get("batch_size", 32))
    force = data.get("force", False)  # Re-detect even if already detected

    if not series_name:
        return jsonify({"error": "series is required"}), 400

    job_id = f"detect_{int(time.time())}"
    _job_queues[job_id] = queue.Queue()
    _job_logs[job_id] = []
    _jobs[job_id] = {"type": "detect", "status": "running", "series": series_name, "progress": None, "last_status": "", "page": "/detect"}

    def run_detect():
        try:
            config = get_app_config()
            conn = init_db(config.db_path)
            try:
                row = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
                if not row:
                    _job_put(job_id, "error", f"Series '{series_name}' not found")
                    return

                _job_put(job_id, "status", "Waiting for GPU access...")
                _gpu_lock.acquire()
                _job_put(job_id, "status", "GPU acquired. Loading models...")

                from vaction.detectors.registry import DetectorRegistry
                registry = DetectorRegistry()
                detector_names = [d.strip() for d in detectors.split(",")]
                dev = config.device

                _job_put(job_id, "status", f"Loading models: {', '.join(detector_names)}...")

                for name in detector_names:
                    if name == "yolo":
                        from vaction.detectors.yolo import YOLODetector
                        registry.register(YOLODetector(model_name=config.yolo_model, device=dev, confidence=config.confidence_threshold))
                    elif name == "clip":
                        from vaction.detectors.clip import CLIPDetector
                        # Load custom prompts from database
                        seed_default_clip_prompts(conn)
                        custom_prompts = get_clip_prompts(conn, enabled_only=True)
                        registry.register(CLIPDetector(
                            model_name=config.clip_model, pretrained=config.clip_pretrained,
                            device=dev, custom_prompts=custom_prompts if custom_prompts else None,
                        ))
                    elif name == "face":
                        from vaction.detectors.faces import FaceDetector
                        det = FaceDetector(device=dev, distance_threshold=config.face_distance_threshold)
                        det.load_character_embeddings(conn, row["id"])
                        registry.register(det)

                _job_put(job_id, "status", "Models loaded. Querying frames...")

                series_id = row["id"]

                # If force mode, clear existing detections and progress for this series
                if force:
                    _job_put(job_id, "status", "Force mode: clearing existing detections...")
                    conn.execute("""
                        DELETE FROM detection WHERE id IN (
                            SELECT d.id FROM detection d
                            JOIN frame f ON d.frame_id = f.id
                            JOIN episode e ON f.episode_id = e.id
                            JOIN season s ON e.season_id = s.id
                            JOIN series sr ON s.series_id = sr.id
                            WHERE sr.name = ? AND d.detector IN ({})
                        )
                    """.format(",".join("?" * len(detector_names))),
                        (series_name, *detector_names),
                    )
                    for dn in detector_names:
                        clear_detection_progress(conn, series_id, dn)
                    conn.commit()

                # Count total frames for this series
                total_all = conn.execute("""
                    SELECT COUNT(*) as c FROM frame f
                    JOIN episode e ON f.episode_id = e.id
                    JOIN season s ON e.season_id = s.id
                    WHERE s.series_id = ?
                """, (series_id,)).fetchone()["c"]

                # Count frames with file paths
                total_with_files = conn.execute("""
                    SELECT COUNT(*) as c FROM frame f
                    JOIN episode e ON f.episode_id = e.id
                    JOIN season s ON e.season_id = s.id
                    WHERE s.series_id = ? AND f.file_path IS NOT NULL
                """, (series_id,)).fetchone()["c"]

                _job_put(job_id, "status", f"Series '{series_name}': {total_all} total frames, {total_with_files} with files")

                # --- Per-detector processing with incremental CLIP support ---
                total_detections = 0
                total_frames_processed = 0

                for detector in registry._detectors:
                    detector_name = detector.name

                    if detector_name == "clip" and not force:
                        # Incremental CLIP: only run new/changed prompts
                        enabled_prompts = get_clip_prompts(conn, enabled_only=True)
                        enabled_labels = {p["label"] for p in enabled_prompts}

                        # Get CLIP labels already detected for this series
                        existing_clip_labels = set(r["label"] for r in conn.execute("""
                            SELECT DISTINCT d.label FROM detection d
                            JOIN frame f ON d.frame_id = f.id
                            JOIN episode e ON f.episode_id = e.id
                            JOIN season s ON e.season_id = s.id
                            WHERE s.series_id = ? AND d.detector = 'clip'
                        """, (series_id,)).fetchall())

                        new_prompts = [p for p in enabled_prompts if p["label"] not in existing_clip_labels]

                        if not new_prompts:
                            _job_put(job_id, "status", f"CLIP: all {len(enabled_prompts)} prompts already detected. Skipping.")
                            continue

                        _job_put(job_id, "status",
                            f"CLIP incremental: {len(new_prompts)} new prompts to detect "
                            f"(skipping {len(existing_clip_labels)} existing)")

                        # Create a new CLIP detector with only the new prompts
                        from vaction.detectors.clip import CLIPDetector
                        clip_detector = CLIPDetector(
                            model_name=config.clip_model, pretrained=config.clip_pretrained,
                            device=dev, custom_prompts=new_prompts,
                        )

                        # For incremental CLIP, process ALL frames (new prompts on all frames)
                        all_frames = get_all_series_frames(conn, series_id)
                        if not all_frames:
                            _job_put(job_id, "status", "CLIP: no frames with file paths found.")
                            continue

                        clip_total = len(all_frames)
                        _job_put(job_id, "status",
                            f"CLIP incremental: running {len(new_prompts)} new prompts on {clip_total} frames")

                        for i in range(0, clip_total, batch_size):
                            batch = all_frames[i:i + batch_size]
                            batch_dicts = [{"id": f["id"], "file_path": f["file_path"]} for f in batch]
                            width = batch[0]["width"]
                            height = batch[0]["height"]

                            count = registry.run_single_detector_on_batch(
                                conn, clip_detector, batch_dicts, width, height)
                            total_detections += count

                            # Mark progress for new CLIP prompts
                            batch_frame_ids = [f["id"] for f in batch]
                            mark_frames_detected_batch(conn, series_id, "clip", batch_frame_ids)
                            conn.commit()

                            processed = min(i + batch_size, clip_total)
                            total_frames_processed = processed
                            _job_put(job_id, "progress", json.dumps({
                                "processed": processed,
                                "total": clip_total,
                                "detections": total_detections,
                                "pct": round(processed / clip_total * 100, 1),
                                "detector": "clip",
                            }))

                    else:
                        # YOLO/face/force-CLIP: use progress tracking to skip already-processed frames
                        if force:
                            frames = get_all_series_frames(conn, series_id)
                        else:
                            frames = get_unprocessed_frames(conn, series_id, detector_name)

                        if not frames:
                            _job_put(job_id, "status", f"{detector_name}: no unprocessed frames. Skipping.")
                            continue

                        det_total = len(frames)
                        _job_put(job_id, "status", f"{detector_name}: processing {det_total} frames...")

                        for i in range(0, det_total, batch_size):
                            batch = frames[i:i + batch_size]
                            batch_dicts = [{"id": f["id"], "file_path": f["file_path"]} for f in batch]
                            width = batch[0]["width"]
                            height = batch[0]["height"]

                            count = registry.run_single_detector_on_batch(
                                conn, detector, batch_dicts, width, height)
                            total_detections += count

                            # Mark progress
                            batch_frame_ids = [f["id"] for f in batch]
                            mark_frames_detected_batch(conn, series_id, detector_name, batch_frame_ids)
                            conn.commit()

                            processed = min(i + batch_size, det_total)
                            _job_put(job_id, "progress", json.dumps({
                                "processed": processed,
                                "total": det_total,
                                "detections": total_detections,
                                "pct": round(processed / det_total * 100, 1),
                                "detector": detector_name,
                            }))

                        total_frames_processed += det_total

                if total_detections == 0 and total_frames_processed == 0:
                    msg = "No unprocessed frames found."
                    if total_with_files == 0:
                        msg += f" {total_all} frames exist but none have file paths. Frame files may have been deleted."
                    else:
                        msg += " All frames already detected for all requested detectors. Enable 'Force re-detect' to re-run."
                    _job_put(job_id, "status", msg)

                _job_put(job_id, "complete", json.dumps({
                    "total_detections": total_detections,
                    "frames_processed": total_frames_processed,
                }))
            finally:
                conn.close()
                _gpu_lock.release()
        except Exception as e:
            import traceback
            _job_put(job_id, "error", f"{e}\n{traceback.format_exc()}")
            try:
                _gpu_lock.release()
            except RuntimeError:
                pass
        finally:
            if _jobs[job_id]["status"] == "running":
                _jobs[job_id]["status"] = "done"

    threading.Thread(target=run_detect, daemon=True).start()
    return jsonify({"job_id": job_id})


# ── API: SSE stream for job progress ────────────────────────────────────────


@app.route("/api/jobs/<job_id>/stream")
def job_stream(job_id):
    if job_id not in _job_queues:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        q = _job_queues[job_id]
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"event: {msg['event']}\ndata: {msg['data']}\n\n"
                if msg["event"] in ("complete", "error"):
                    break
            except queue.Empty:
                yield "event: ping\ndata: keepalive\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/jobs")
def api_jobs():
    return jsonify({jid: {"type": j["type"], "status": j["status"], "series": j.get("series")} for jid, j in _jobs.items()})


@app.route("/api/jobs/active")
def api_jobs_active():
    """Return all currently running jobs with progress info — polled by global status bar."""
    active = []
    for jid, j in _jobs.items():
        if j["status"] in ("running",):
            active.append({
                "job_id": jid,
                "type": j["type"],
                "series": j.get("series", ""),
                "page": j.get("page", ""),
                "last_status": j.get("last_status", ""),
                "progress": j.get("progress"),
            })
    # Also include recently completed (last 10 seconds) so UI can show "done"
    for jid, j in _jobs.items():
        if j["status"] in ("done", "error") and j.get("progress"):
            active.append({
                "job_id": jid,
                "type": j["type"],
                "series": j.get("series", ""),
                "page": j.get("page", ""),
                "status": j["status"],
                "last_status": j.get("last_status", ""),
                "progress": j.get("progress"),
                "result": j.get("result"),
                "error": j.get("error"),
            })
    return jsonify({"jobs": active})


# ── API: Search ──────────────────────────────────────────────────────────────


@app.route("/api/search")
def api_search():
    query_str = request.args.get("q", "").strip()
    series_name = request.args.get("series", "").strip()
    confidence = float(request.args.get("confidence", 0.25))

    if not query_str or not series_name:
        return jsonify({"error": "q and series are required"}), 400

    conn = get_db()
    try:
        row = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not row:
            return jsonify({"error": f"Series '{series_name}' not found"}), 404

        from vaction.metrics import compute_episode_metrics
        # Expand theme names into OR queries
        query_expanded = _expand_themes(conn, query_str)
        results = compute_episode_metrics(conn, query_expanded, confidence, series_id=row["id"])

        from vaction.metrics import format_timestamp
        output = []
        for r in results:
            output.append({
                "season": r.season_number,
                "episode": r.episode_number,
                "title": r.episode_title,
                "episode_share_pct": round(r.episode_share_pct, 4),
                "pixel_time": round(r.pixel_time, 2),
                "pixel_minutes": round(r.pixel_minutes, 2),
                "strongest_windows": [
                    {"start": format_timestamp(s), "end": format_timestamp(e), "start_s": round(s, 1), "end_s": round(e, 1)}
                    for s, e in r.strongest_windows
                ],
            })

        return jsonify({"query": query_str, "series": series_name, "results": output})
    finally:
        conn.close()


@app.route("/api/labels")
def api_labels():
    """Get all unique detection labels in the database."""
    series_name = request.args.get("series", "").strip()
    conn = get_db()
    try:
        if series_name:
            rows = conn.execute("""
                SELECT DISTINCT d.label, d.detector, COUNT(*) as count
                FROM detection d
                JOIN frame f ON d.frame_id = f.id
                JOIN episode e ON f.episode_id = e.id
                JOIN season s ON e.season_id = s.id
                JOIN series sr ON s.series_id = sr.id
                WHERE sr.name = ?
                GROUP BY d.label, d.detector
                ORDER BY count DESC
            """, (series_name,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT DISTINCT label, detector, COUNT(*) as count
                FROM detection GROUP BY label, detector ORDER BY count DESC
            """).fetchall()

        return jsonify({"labels": [{"label": r["label"], "detector": r["detector"], "count": r["count"]} for r in rows]})
    finally:
        conn.close()


@app.route("/api/series")
def api_series():
    conn = get_db()
    try:
        rows = conn.execute("SELECT id, name FROM series ORDER BY name").fetchall()
        return jsonify({"series": [{"id": r["id"], "name": r["name"]} for r in rows]})
    finally:
        conn.close()


@app.route("/api/alias", methods=["POST"])
def api_alias():
    data = request.json
    term = data.get("term", "").strip()
    labels = data.get("labels", [])
    if not term or not labels:
        return jsonify({"error": "term and labels are required"}), 400

    conn = get_db()
    try:
        for label in labels:
            insert_alias(conn, term, label.strip())
        return jsonify({"ok": True, "term": term, "labels": labels})
    finally:
        conn.close()


# ── API: Chart data ─────────────────────────────────────────────────────────


@app.route("/api/chart")
def api_chart():
    """Return data suitable for charting: episode share across episodes for one or more terms.
    Optional params: episodes (comma-separated e.g. S01E01,S01E02) to filter.
    """
    series_name = request.args.get("series", "").strip()
    terms = request.args.get("terms", "").strip()  # comma-separated
    confidence = float(request.args.get("confidence", 0.25))
    episode_filter = request.args.get("episodes", "").strip()  # comma-separated S01E01 codes

    if not series_name or not terms:
        return jsonify({"error": "series and terms are required"}), 400

    conn = get_db()
    try:
        row = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not row:
            return jsonify({"error": f"Series '{series_name}' not found"}), 404

        from vaction.metrics import compute_episode_metrics

        # Get all episodes for the x-axis
        episodes = conn.execute("""
            SELECT e.number as ep_num, s.number as season_num, e.title, e.id as ep_id
            FROM episode e
            JOIN season s ON e.season_id = s.id
            WHERE s.series_id = ?
            ORDER BY s.number, e.number
        """, (row["id"],)).fetchall()

        ep_labels = [f"S{e['season_num']:02d}E{e['ep_num']:02d}" for e in episodes]

        # Apply episode filter
        if episode_filter:
            filter_set = {e.strip().upper() for e in episode_filter.split(",") if e.strip()}
            filtered_indices = [i for i, lbl in enumerate(ep_labels) if lbl in filter_set]
            ep_labels = [ep_labels[i] for i in filtered_indices]
        else:
            filtered_indices = list(range(len(ep_labels)))

        term_list = [t.strip() for t in terms.split(",") if t.strip()]
        datasets = []

        for term in term_list:
            expanded_term = _expand_themes(conn, term)
            results = compute_episode_metrics(conn, expanded_term, confidence, series_id=row["id"])
            # Build a map of episode -> share
            ep_map = {}
            for r in results:
                key = f"S{r.season_number:02d}E{r.episode_number:02d}"
                ep_map[key] = round(r.episode_share_pct, 4)

            data_points = [ep_map.get(ep_labels[j], 0) for j in range(len(ep_labels))]
            datasets.append({"term": term, "data": data_points})

        return jsonify({"labels": ep_labels, "datasets": datasets})
    finally:
        conn.close()


@app.route("/api/episodes")
def api_episodes():
    """Return all episodes for a series (for chart episode picker)."""
    series_name = request.args.get("series", "").strip()
    if not series_name:
        return jsonify({"episodes": []})
    conn = get_db()
    try:
        row = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not row:
            return jsonify({"episodes": []})
        episodes = conn.execute("""
            SELECT e.number as ep_num, s.number as season_num, e.title
            FROM episode e
            JOIN season s ON e.season_id = s.id
            WHERE s.series_id = ?
            ORDER BY s.number, e.number
        """, (row["id"],)).fetchall()
        return jsonify({"episodes": [
            {"code": f"S{e['season_num']:02d}E{e['ep_num']:02d}", "title": e["title"] or ""}
            for e in episodes
        ]})
    finally:
        conn.close()


def _expand_themes(conn, query_str: str) -> str:
    """If a query term matches a theme name, expand it to (tag1 OR tag2 OR ...)."""
    themes = get_all_themes(conn)
    theme_map = {t["name"].lower(): t["labels"] for t in themes}

    # Simple token-level replacement
    import re
    tokens = re.findall(r'"[^"]*"|\(|\)|AND|OR|NOT|[^\s()]+', query_str, re.IGNORECASE)
    result = []
    for token in tokens:
        lower = token.lower()
        if lower in theme_map and lower not in ("and", "or", "not"):
            labels = theme_map[lower]
            if labels:
                expanded = " OR ".join(labels)
                result.append(f"({expanded})")
            else:
                result.append(token)
        else:
            result.append(token)
    return " ".join(result)


@app.route("/api/chart/breakdown")
def api_chart_breakdown():
    """Return per-label episode share data for each term.
    If a term is a theme, break it down into its constituent labels.
    Returns: {episodes: [...], series: [{name, labels: [{label, data: [...]}]}]}
    """
    series_name = request.args.get("series", "").strip()
    terms = request.args.get("terms", "").strip()
    confidence = float(request.args.get("confidence", 0.25))
    episode_filter = request.args.get("episodes", "").strip()

    if not series_name or not terms:
        return jsonify({"error": "series and terms are required"}), 400

    conn = get_db()
    try:
        row = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not row:
            return jsonify({"error": f"Series '{series_name}' not found"}), 404
        series_id = row["id"]

        from vaction.metrics import compute_episode_metrics

        episodes = conn.execute("""
            SELECT e.number as ep_num, s.number as season_num, e.title
            FROM episode e JOIN season s ON e.season_id = s.id
            WHERE s.series_id = ? ORDER BY s.number, e.number
        """, (series_id,)).fetchall()

        ep_labels = [f"S{e['season_num']:02d}E{e['ep_num']:02d}" for e in episodes]

        if episode_filter:
            filter_set = {e.strip().upper() for e in episode_filter.split(",") if e.strip()}
            ep_labels = [lbl for lbl in ep_labels if lbl in filter_set]

        # Get theme map for breakdown
        themes = get_all_themes(conn)
        theme_map = {t["name"].lower(): t["labels"] for t in themes}

        term_list = [t.strip() for t in terms.split(",") if t.strip()]
        groups = []  # each group = {name: str, labels: [{label, data: [...]}]}

        for term in term_list:
            lower = term.lower()
            if lower in theme_map and theme_map[lower]:
                # Theme: break into individual labels
                sub_labels = theme_map[lower]
                group = {"name": term, "labels": []}
                for label in sub_labels:
                    results = compute_episode_metrics(conn, label, confidence, series_id=series_id)
                    ep_map = {f"S{r.season_number:02d}E{r.episode_number:02d}": round(r.episode_share_pct, 4) for r in results}
                    group["labels"].append({"label": label, "data": [ep_map.get(ep, 0) for ep in ep_labels]})
                groups.append(group)
            else:
                # Single label
                expanded = _expand_themes(conn, term)
                results = compute_episode_metrics(conn, expanded, confidence, series_id=series_id)
                ep_map = {f"S{r.season_number:02d}E{r.episode_number:02d}": round(r.episode_share_pct, 4) for r in results}
                groups.append({"name": term, "labels": [{"label": term, "data": [ep_map.get(ep, 0) for ep in ep_labels]}]})

        return jsonify({"episodes": ep_labels, "groups": groups})
    finally:
        conn.close()


@app.route("/api/insights")
def api_insights():
    """Auto-generate insights for a series: top labels, trends, anomalies."""
    series_name = request.args.get("series", "").strip()
    if not series_name:
        return jsonify({"error": "series is required"}), 400

    conn = get_db()
    try:
        row = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not row:
            return jsonify({"error": f"Series '{series_name}' not found"}), 404
        series_id = row["id"]

        from vaction.metrics import compute_episode_metrics

        # Get all labels with detection counts
        labels_data = conn.execute("""
            SELECT d.label, COUNT(*) as count, AVG(d.confidence) as avg_conf,
                   COUNT(DISTINCT f.episode_id) as episode_count
            FROM detection d
            JOIN frame f ON d.frame_id = f.id
            JOIN episode e ON f.episode_id = e.id
            JOIN season s ON e.season_id = s.id
            WHERE s.series_id = ? AND d.label != 'unknown_face'
            GROUP BY d.label
            ORDER BY count DESC
        """, (series_id,)).fetchall()

        # Get episode list
        episodes = conn.execute("""
            SELECT e.id, e.number as ep_num, s.number as season_num, e.title,
                   e.num_frames, e.width, e.height
            FROM episode e JOIN season s ON e.season_id = s.id
            WHERE s.series_id = ? ORDER BY s.number, e.number
        """, (series_id,)).fetchall()

        total_episodes = len(episodes)
        ep_codes = [f"S{e['season_num']:02d}E{e['ep_num']:02d}" for e in episodes]

        # Top 20 labels by detection count
        top_labels = [{"label": r["label"], "count": r["count"],
                       "avg_confidence": round(r["avg_conf"], 3),
                       "episode_count": r["episode_count"],
                       "coverage": round(r["episode_count"] / max(total_episodes, 1) * 100, 1)}
                      for r in labels_data[:20]]

        # Compute episode shares for top 10 labels to find trends
        trending = []
        declining = []
        spiky = []
        for label_info in labels_data[:15]:
            label = label_info["label"]
            results = compute_episode_metrics(conn, label, 0.25, series_id=series_id)
            ep_map = {f"S{r.season_number:02d}E{r.episode_number:02d}": r.episode_share_pct for r in results}
            shares = [ep_map.get(ep, 0) for ep in ep_codes]

            if len(shares) >= 3:
                first_half = sum(shares[:len(shares)//2])
                second_half = sum(shares[len(shares)//2:])
                if second_half > first_half * 1.5 and second_half > 0.001:
                    trending.append({"label": label, "change": round((second_half - first_half) / max(first_half, 0.0001) * 100, 1), "shares": [round(s, 4) for s in shares]})
                elif first_half > second_half * 1.5 and first_half > 0.001:
                    declining.append({"label": label, "change": round((first_half - second_half) / max(second_half, 0.0001) * 100, 1), "shares": [round(s, 4) for s in shares]})

                avg = sum(shares) / len(shares) if shares else 0
                if avg > 0:
                    max_share = max(shares)
                    if max_share > avg * 3:
                        peak_ep = ep_codes[shares.index(max_share)]
                        spiky.append({"label": label, "peak_episode": peak_ep, "peak_value": round(max_share, 4), "avg_value": round(avg, 4)})

        # Episode complexity (number of unique labels per episode)
        ep_complexity = []
        for ep in episodes:
            unique_labels = conn.execute("""
                SELECT COUNT(DISTINCT d.label) as c FROM detection d
                JOIN frame f ON d.frame_id = f.id
                WHERE f.episode_id = ?
            """, (ep["id"],)).fetchone()["c"]
            ep_complexity.append({
                "code": f"S{ep['season_num']:02d}E{ep['ep_num']:02d}",
                "title": ep["title"] or "",
                "unique_labels": unique_labels,
            })

        return jsonify({
            "series": series_name,
            "total_episodes": total_episodes,
            "total_labels": len(labels_data),
            "episodes": ep_codes,
            "top_labels": top_labels,
            "trending": sorted(trending, key=lambda x: -x["change"])[:5],
            "declining": sorted(declining, key=lambda x: -x["change"])[:5],
            "spiky": sorted(spiky, key=lambda x: -x["peak_value"])[:5],
            "episode_complexity": ep_complexity,
        })
    finally:
        conn.close()


# ── API: File browser ────────────────────────────────────────────────────────


@app.route("/api/browse")
def api_browse():
    """Browse local filesystem directories."""
    path = request.args.get("path", "").strip()
    if not path:
        path = str(Path.home())

    p = Path(path).expanduser().resolve()
    if not p.exists():
        return jsonify({"error": f"Path does not exist: {p}", "path": str(p)}), 404

    if p.is_file():
        p = p.parent

    entries = []
    try:
        for item in sorted(p.iterdir()):
            if item.name.startswith("."):
                continue
            entries.append({
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else None,
            })
    except PermissionError:
        return jsonify({"error": "Permission denied", "path": str(p)}), 403

    parent = str(p.parent) if p != p.parent else None
    return jsonify({"path": str(p), "parent": parent, "entries": entries})


# ── API: Vocabulary (CLIP prompts) ──────────────────────────────────────────


@app.route("/api/vocabulary")
def api_vocabulary():
    conn = get_db()
    try:
        seed_default_clip_prompts(conn)
        prompts = get_clip_prompts(conn, enabled_only=False)
        return jsonify({"prompts": prompts})
    finally:
        conn.close()


@app.route("/api/vocabulary", methods=["POST"])
def api_vocabulary_add():
    data = request.json
    prompt = data.get("prompt", "").strip()
    label = data.get("label", "").strip()
    category = data.get("category", "scene").strip()
    if not prompt or not label:
        return jsonify({"error": "prompt and label are required"}), 400

    conn = get_db()
    try:
        pid = add_clip_prompt(conn, prompt, label, category)
        return jsonify({"ok": True, "id": pid})
    finally:
        conn.close()


@app.route("/api/vocabulary/<int:prompt_id>", methods=["DELETE"])
def api_vocabulary_delete(prompt_id):
    conn = get_db()
    try:
        delete_clip_prompt(conn, prompt_id)
        return jsonify({"ok": True})
    finally:
        conn.close()


@app.route("/api/vocabulary/<int:prompt_id>/toggle", methods=["POST"])
def api_vocabulary_toggle(prompt_id):
    data = request.json
    enabled = data.get("enabled", True)
    conn = get_db()
    try:
        toggle_clip_prompt(conn, prompt_id, enabled)
        return jsonify({"ok": True})
    finally:
        conn.close()


@app.route("/api/vocabulary/preset", methods=["POST"])
def api_vocabulary_load_preset():
    """Load a vocabulary preset (low=1, medium=2, high=3). Replaces all existing prompts."""
    data = request.json
    level = int(data.get("level", 2))
    if level not in (1, 2, 3):
        return jsonify({"error": "level must be 1, 2, or 3"}), 400
    conn = get_db()
    try:
        count = load_vocabulary_preset(conn, level)
        from vaction.vocabulary_presets import get_level_counts
        counts = get_level_counts()
        return jsonify({"ok": True, "loaded": count, "counts": counts})
    finally:
        conn.close()


@app.route("/api/vocabulary/counts")
def api_vocabulary_counts():
    """Return counts for each preset level."""
    from vaction.vocabulary_presets import get_level_counts
    return jsonify(get_level_counts())


# ── API: Themes ─────────────────────────────────────────────────────────────


@app.route("/api/themes")
def api_themes():
    conn = get_db()
    try:
        seed_default_themes(conn)
        themes = get_all_themes(conn)
        return jsonify({"themes": themes})
    finally:
        conn.close()


@app.route("/api/themes", methods=["POST"])
def api_themes_create():
    data = request.json
    name = data.get("name", "").strip()
    description = data.get("description", "").strip()
    labels = data.get("labels", [])
    if not name or not labels:
        return jsonify({"error": "name and labels are required"}), 400

    conn = get_db()
    try:
        tid = create_theme(conn, name, description, labels)
        return jsonify({"ok": True, "id": tid})
    finally:
        conn.close()


@app.route("/api/themes/<int:theme_id>", methods=["PUT"])
def api_themes_update(theme_id):
    data = request.json
    name = data.get("name", "").strip()
    description = data.get("description", "").strip()
    labels = data.get("labels", [])
    if not name:
        return jsonify({"error": "name is required"}), 400

    conn = get_db()
    try:
        update_theme(conn, theme_id, name, description, labels)
        return jsonify({"ok": True})
    finally:
        conn.close()


@app.route("/api/themes/<int:theme_id>", methods=["DELETE"])
def api_themes_delete(theme_id):
    conn = get_db()
    try:
        delete_theme(conn, theme_id)
        return jsonify({"ok": True})
    finally:
        conn.close()


@app.route("/api/themes/reset", methods=["POST"])
def api_themes_reset():
    """Delete all themes and re-seed from defaults."""
    conn = get_db()
    try:
        # Delete all existing themes
        themes = get_all_themes(conn)
        for t in themes:
            delete_theme(conn, t["id"])
        # Force re-seed
        from vaction.vocabulary_presets import DEFAULT_THEMES
        created = 0
        for t in DEFAULT_THEMES:
            create_theme(conn, t["name"], t["description"], t["labels"])
            created += 1
        return jsonify({"ok": True, "created": created})
    finally:
        conn.close()


# ── API: Faces ──────────────────────────────────────────────────────────────


@app.route("/api/faces")
def api_faces():
    """Get all face clusters for a series."""
    series_name = request.args.get("series", "").strip()
    conn = get_db()
    try:
        if not series_name:
            return jsonify({"clusters": []})

        row = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not row:
            return jsonify({"clusters": []})

        clusters = conn.execute("""
            SELECT fc.id, fc.name, fc.thumbnail_frame_id, fc.thumbnail_bbox,
                   COUNT(fcm.id) as member_count
            FROM face_cluster fc
            LEFT JOIN face_cluster_member fcm ON fcm.cluster_id = fc.id
            WHERE fc.series_id = ?
            GROUP BY fc.id
            ORDER BY member_count DESC
        """, (row["id"],)).fetchall()

        result = []
        for c in clusters:
            result.append({
                "id": c["id"],
                "name": c["name"] or f"Face #{c['id']}",
                "member_count": c["member_count"],
                "thumbnail_frame_id": c["thumbnail_frame_id"],
                "thumbnail_bbox": json.loads(c["thumbnail_bbox"]) if c["thumbnail_bbox"] else None,
            })

        return jsonify({"clusters": result})
    finally:
        conn.close()


@app.route("/api/faces/scan", methods=["POST"])
def api_faces_scan():
    """Scan for faces, cluster them, and store in face_cluster tables."""
    data = request.json
    series_name = data.get("series", "").strip()
    if not series_name:
        return jsonify({"error": "series is required"}), 400
    resolution = data.get("resolution")
    if resolution is not None:
        resolution = int(resolution)

    job_id = f"facescan_{int(time.time())}"
    _job_queues[job_id] = queue.Queue()
    _job_logs[job_id] = []
    _jobs[job_id] = {"type": "facescan", "status": "running", "series": series_name, "progress": None, "last_status": "", "page": "/faces"}

    def run_facescan():
        try:
            import numpy as np
            import cv2
            from collections import defaultdict
            from concurrent.futures import ThreadPoolExecutor
            config = get_app_config()

            _job_put(job_id, "status", "Waiting for GPU access...")
            _gpu_lock.acquire()
            _job_put(job_id, "status", "GPU acquired. Starting face scan...")

            conn = init_db(config.db_path)
            try:
                row = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
                if not row:
                    _job_put(job_id, "error", f"Series '{series_name}' not found")
                    return
                series_id = row["id"]

                # Get all face detections
                face_dets = conn.execute("""
                    SELECT d.id as det_id, d.frame_id, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
                           d.confidence, f.file_path, e.width, e.height
                    FROM detection d
                    JOIN frame f ON d.frame_id = f.id
                    JOIN episode e ON f.episode_id = e.id
                    JOIN season s ON e.season_id = s.id
                    WHERE s.series_id = ?
                      AND (d.detector = 'face' OR d.label IN ('face', 'person'))
                      AND d.bbox_x IS NOT NULL
                      AND f.file_path IS NOT NULL
                    ORDER BY d.confidence DESC
                """, (series_id,)).fetchall()

                if not face_dets:
                    _job_put(job_id, "status", "No face detections found. Run detection with 'Faces only' or 'All' detectors first, then come back here to cluster.")
                    _job_put(job_id, "complete", json.dumps({"clusters": 0}))
                    return

                _job_put(job_id, "status", f"Found {len(face_dets)} face detections. Computing embeddings...")

                # Extract face embeddings using insightface
                from vaction.detectors.faces import FaceDetector
                detector = FaceDetector(device=config.device)
                detector._ensure_model()

                # Group detections by frame to avoid redundant image loads
                frame_groups = defaultdict(list)
                for det in face_dets:
                    if det["file_path"] and det["bbox_x"] is not None:
                        frame_groups[det["frame_id"]].append(det)

                # Pre-load frame images in parallel (I/O-bound)
                frame_paths = {fid: dets[0]["file_path"] for fid, dets in frame_groups.items()}

                def load_frame(file_path):
                    return cv2.imread(file_path)

                _job_put(job_id, "status", f"Loading {len(frame_paths)} unique frames...")
                with ThreadPoolExecutor(max_workers=8) as pool:
                    loaded = dict(zip(frame_paths.keys(), pool.map(load_frame, frame_paths.values())))

                embeddings = []
                det_ids = []
                det_infos = []

                # Process one frame at a time: run InsightFace once per frame
                for i, (frame_id, dets) in enumerate(frame_groups.items()):
                    img = loaded.get(frame_id)
                    if img is None:
                        continue

                    # Optionally resize for speed
                    scale = 1.0
                    if resolution and resolution < img.shape[0]:
                        scale = resolution / img.shape[0]
                        img = cv2.resize(img, None, fx=scale, fy=scale)

                    try:
                        faces = detector._app.get(img)
                    except Exception:
                        continue

                    # Match each detection in this frame to the best InsightFace result
                    for det in dets:
                        best_face = None
                        best_overlap = 0
                        # Scale bbox coords if we resized
                        bx = det["bbox_x"] * scale
                        by = det["bbox_y"] * scale
                        bw = det["bbox_w"] * scale
                        bh = det["bbox_h"] * scale
                        for face in faces:
                            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                            ox = max(0, min(fx2, bx + bw) - max(fx1, bx))
                            oy = max(0, min(fy2, by + bh) - max(fy1, by))
                            overlap = ox * oy
                            if overlap > best_overlap and face.embedding is not None:
                                best_overlap = overlap
                                best_face = face

                        if best_face is not None and best_face.embedding is not None:
                            embeddings.append(best_face.embedding.astype(np.float32))
                            det_ids.append(det["det_id"])
                            det_infos.append(det)

                    if (i + 1) % 20 == 0:
                        _job_put(job_id, "progress", json.dumps({
                            "processed": i + 1, "total": len(frame_groups),
                            "pct": round((i + 1) / len(frame_groups) * 100, 1),
                            "embeddings": len(embeddings),
                        }))

                if not embeddings:
                    _job_put(job_id, "complete", json.dumps({"clusters": 0}))
                    return

                _job_put(job_id, "status", f"Got {len(embeddings)} embeddings. Clustering...")

                # Simple agglomerative clustering by cosine similarity
                emb_matrix = np.stack(embeddings)
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                emb_norm = emb_matrix / (norms + 1e-8)
                sim_matrix = emb_norm @ emb_norm.T

                threshold = 0.55  # cosine similarity threshold
                assigned = [-1] * len(embeddings)
                cluster_id_counter = 0

                for i in range(len(embeddings)):
                    if assigned[i] >= 0:
                        continue
                    assigned[i] = cluster_id_counter
                    for j in range(i + 1, len(embeddings)):
                        if assigned[j] >= 0:
                            continue
                        if sim_matrix[i, j] >= threshold:
                            assigned[j] = cluster_id_counter
                    cluster_id_counter += 1

                # Clear existing clusters for this series
                existing = conn.execute("SELECT id FROM face_cluster WHERE series_id = ?", (series_id,)).fetchall()
                for ec in existing:
                    conn.execute("DELETE FROM face_cluster_member WHERE cluster_id = ?", (ec["id"],))
                    conn.execute("DELETE FROM face_cluster WHERE id = ?", (ec["id"],))

                # Create clusters
                cluster_map = {}
                for idx, cid in enumerate(assigned):
                    if cid not in cluster_map:
                        # Use the first (highest confidence) face as thumbnail
                        det = det_infos[idx]
                        bbox_json = json.dumps({"x": det["bbox_x"], "y": det["bbox_y"], "w": det["bbox_w"], "h": det["bbox_h"]})
                        cur = conn.execute(
                            "INSERT INTO face_cluster (series_id, name, thumbnail_frame_id, thumbnail_bbox) VALUES (?, ?, ?, ?)",
                            (series_id, None, det["frame_id"], bbox_json),
                        )
                        cluster_map[cid] = cur.lastrowid

                    conn.execute(
                        "INSERT OR IGNORE INTO face_cluster_member (cluster_id, detection_id, embedding) VALUES (?, ?, ?)",
                        (cluster_map[cid], det_ids[idx], embeddings[idx].tobytes()),
                    )

                conn.commit()
                _job_put(job_id, "complete", json.dumps({"clusters": len(cluster_map), "faces": len(embeddings)}))
            finally:
                conn.close()
                _gpu_lock.release()
        except Exception as e:
            import traceback
            _job_put(job_id, "error", f"{e}\n{traceback.format_exc()}")
            try:
                _gpu_lock.release()
            except RuntimeError:
                pass
        finally:
            if _jobs[job_id]["status"] == "running":
                _jobs[job_id]["status"] = "done"

    threading.Thread(target=run_facescan, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/faces/<int:cluster_id>/name", methods=["POST"])
def api_faces_rename(cluster_id):
    data = request.json
    name = data.get("name", "").strip()
    conn = get_db()
    try:
        conn.execute("UPDATE face_cluster SET name = ? WHERE id = ?", (name, cluster_id))
        conn.commit()
        return jsonify({"ok": True})
    finally:
        conn.close()


@app.route("/api/faces/<int:cluster_id>/chart")
def api_faces_chart(cluster_id):
    """Get episode-level data for a face cluster, similar to search chart."""
    series_name = request.args.get("series", "").strip()
    conn = get_db()
    try:
        # Get detection IDs in this cluster
        members = conn.execute(
            "SELECT detection_id FROM face_cluster_member WHERE cluster_id = ?", (cluster_id,)
        ).fetchall()
        det_ids = [m["detection_id"] for m in members]

        if not det_ids:
            return jsonify({"labels": [], "data": []})

        # Get episode-level aggregation
        placeholders = ",".join("?" * len(det_ids))
        rows = conn.execute(f"""
            SELECT s.number as season_num, e.number as ep_num,
                   SUM(d.pixel_area) as total_area,
                   e.num_frames, e.width, e.height
            FROM detection d
            JOIN frame f ON d.frame_id = f.id
            JOIN episode e ON f.episode_id = e.id
            JOIN season s ON e.season_id = s.id
            WHERE d.id IN ({placeholders})
            GROUP BY e.id
            ORDER BY s.number, e.number
        """, det_ids).fetchall()

        labels = []
        data = []
        for r in rows:
            labels.append(f"S{r['season_num']:02d}E{r['ep_num']:02d}")
            budget = r["num_frames"] * r["width"] * r["height"]
            share = (r["total_area"] / budget * 100) if budget > 0 else 0
            data.append(round(share, 4))

        return jsonify({"labels": labels, "data": data})
    finally:
        conn.close()


@app.route("/api/faces/frame/<int:frame_id>")
def api_face_thumbnail(frame_id):
    """Serve a cropped face image from a frame."""
    bbox = request.args.get("bbox", "")
    conn = get_db()
    try:
        row = conn.execute("SELECT file_path FROM frame WHERE id = ?", (frame_id,)).fetchone()
        if not row or not row["file_path"]:
            return "Not found", 404

        from PIL import Image
        import io
        img = Image.open(row["file_path"])

        if bbox:
            b = json.loads(bbox)
            x, y, w, h = b.get("x", 0), b.get("y", 0), b.get("w", img.width), b.get("h", img.height)
            # Add padding
            pad = int(max(w, h) * 0.2)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.width, x + w + pad)
            y2 = min(img.height, y + h + pad)
            img = img.crop((x1, y1, x2, y2))

        img.thumbnail((150, 150))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        return Response(buf.getvalue(), mimetype="image/jpeg")
    finally:
        conn.close()


# ── API: Data persistence info ──────────────────────────────────────────────


@app.route("/api/data-info")
def api_data_info():
    """Return info about the data directory and database."""
    config = get_app_config()
    db_path = config.db_path
    frames_dir = config.frames_dir

    db_size = db_path.stat().st_size if db_path.exists() else 0
    frames_size = sum(f.stat().st_size for f in frames_dir.rglob("*") if f.is_file()) if frames_dir.exists() else 0

    return jsonify({
        "data_dir": str(config.data_dir),
        "db_path": str(db_path),
        "db_size_mb": round(db_size / 1024 / 1024, 2),
        "frames_dir": str(frames_dir),
        "frames_size_mb": round(frames_size / 1024 / 1024, 2),
        "total_size_mb": round((db_size + frames_size) / 1024 / 1024, 2),
    })


@app.route("/api/backup", methods=["POST"])
def api_backup():
    """Create a backup of the database."""
    import shutil
    config = get_app_config()
    db_path = config.db_path
    if not db_path.exists():
        return jsonify({"error": "No database to backup"}), 404

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = config.data_dir / f"vaction_backup_{timestamp}.db"
    shutil.copy2(str(db_path), str(backup_path))
    return jsonify({"ok": True, "backup_path": str(backup_path), "size_mb": round(backup_path.stat().st_size / 1024 / 1024, 2)})


@app.route("/api/backups")
def api_backups():
    config = get_app_config()
    backups = sorted(config.data_dir.glob("vaction_backup_*.db"), reverse=True)
    return jsonify({"backups": [
        {"path": str(b), "name": b.name, "size_mb": round(b.stat().st_size / 1024 / 1024, 2), "modified": b.stat().st_mtime}
        for b in backups
    ]})


# ── API: Explorer ────────────────────────────────────────────────────────────


@app.route("/api/explorer")
def api_explorer():
    """Browse detections with filtering and pagination."""
    series_name = request.args.get("series", "").strip()
    label = request.args.get("label", "").strip()
    detector = request.args.get("detector", "").strip()
    episode = request.args.get("episode", "").strip()
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(200, max(1, int(request.args.get("per_page", 50))))

    if not series_name:
        return jsonify({"error": "series is required"}), 400

    conn = get_db()
    try:
        sr = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not sr:
            return jsonify({"error": f"Series '{series_name}' not found"}), 404

        conditions = ["sn.series_id = ?"]
        params: list = [sr["id"]]

        if label:
            conditions.append("d.label = ?")
            params.append(label)
        if detector and detector != "all":
            conditions.append("d.detector = ?")
            params.append(detector)
        if episode:
            ep_match = re.match(r"S(\d+)E(\d+)", episode, re.IGNORECASE)
            if ep_match:
                conditions.append("sn.number = ? AND e.number = ?")
                params.append(int(ep_match.group(1)))
                params.append(int(ep_match.group(2)))

        where = " AND ".join(conditions)

        count_sql = f"""
            SELECT COUNT(*) as c
            FROM detection d
            JOIN frame f ON d.frame_id = f.id
            JOIN episode e ON f.episode_id = e.id
            JOIN season sn ON e.season_id = sn.id
            WHERE {where}
        """
        total = conn.execute(count_sql, params).fetchone()["c"]

        offset = (page - 1) * per_page
        data_sql = f"""
            SELECT d.id, d.frame_id, d.label, d.detector, d.confidence,
                   d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, d.pixel_area,
                   sn.number as season_num, e.number as ep_num, e.title as episode_title,
                   f.timestamp, f.file_path as frame_path
            FROM detection d
            JOIN frame f ON d.frame_id = f.id
            JOIN episode e ON f.episode_id = e.id
            JOIN season sn ON e.season_id = sn.id
            WHERE {where}
            ORDER BY sn.number, e.number, f.timestamp, d.id
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(data_sql, params + [per_page, offset]).fetchall()

        detections = []
        for r in rows:
            detections.append({
                "id": r["id"],
                "frame_id": r["frame_id"],
                "label": r["label"],
                "detector": r["detector"],
                "confidence": round(r["confidence"], 4),
                "bbox_x": r["bbox_x"],
                "bbox_y": r["bbox_y"],
                "bbox_w": r["bbox_w"],
                "bbox_h": r["bbox_h"],
                "pixel_area": r["pixel_area"],
                "episode_code": f"S{r['season_num']:02d}E{r['ep_num']:02d}",
                "episode_title": r["episode_title"] or "",
                "timestamp": round(r["timestamp"], 2),
                "frame_path": r["frame_path"] or "",
            })

        pages = max(1, (total + per_page - 1) // per_page)
        return jsonify({
            "detections": detections,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": pages,
        })
    finally:
        conn.close()


@app.route("/api/explorer/timeline")
def api_explorer_timeline():
    """Get timeline data for an episode: detection density in 5-second segments."""
    series_name = request.args.get("series", "").strip()
    episode_code = request.args.get("episode", "").strip()

    if not series_name or not episode_code:
        return jsonify({"error": "series and episode are required"}), 400

    ep_match = re.match(r"S(\d+)E(\d+)", episode_code, re.IGNORECASE)
    if not ep_match:
        return jsonify({"error": "Invalid episode format, use S01E03"}), 400

    season_num = int(ep_match.group(1))
    ep_num = int(ep_match.group(2))

    conn = get_db()
    try:
        sr = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not sr:
            return jsonify({"error": f"Series '{series_name}' not found"}), 404

        ep = conn.execute("""
            SELECT e.id, e.duration_secs, e.fps_sampled
            FROM episode e
            JOIN season sn ON e.season_id = sn.id
            WHERE sn.series_id = ? AND sn.number = ? AND e.number = ?
        """, (sr["id"], season_num, ep_num)).fetchone()
        if not ep:
            return jsonify({"error": "Episode not found"}), 404

        duration = ep["duration_secs"]
        fps = ep["fps_sampled"]
        segment_size = 5.0

        dets = conn.execute("""
            SELECT d.label, f.timestamp
            FROM detection d
            JOIN frame f ON d.frame_id = f.id
            WHERE f.episode_id = ?
            ORDER BY f.timestamp
        """, (ep["id"],)).fetchall()

        num_segments = max(1, int(duration / segment_size) + 1)
        segments = []
        label_set = set()

        for i in range(num_segments):
            seg_start = i * segment_size
            seg_end = seg_start + segment_size
            labels: dict[str, int] = {}
            for d in dets:
                if seg_start <= d["timestamp"] < seg_end:
                    lbl = d["label"]
                    labels[lbl] = labels.get(lbl, 0) + 1
                    label_set.add(lbl)
            density = sum(labels.values())
            if density > 0:
                segments.append({
                    "start": round(seg_start, 1),
                    "end": round(seg_end, 1),
                    "labels": labels,
                    "density": density,
                })

        palette = [
            "#4a9eff", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8",
            "#ff922b", "#20c997", "#a9e34b", "#e599f7", "#74c0fc",
            "#f06595", "#66d9e8", "#ffe066", "#c0eb75", "#b197fc",
        ]
        label_colors = {}
        for i, lbl in enumerate(sorted(label_set)):
            label_colors[lbl] = palette[i % len(palette)]

        return jsonify({
            "duration": duration,
            "fps": fps,
            "segments": segments,
            "label_colors": label_colors,
        })
    finally:
        conn.close()


@app.route("/api/explorer/frame/<int:frame_id>")
def api_explorer_frame(frame_id):
    """Serve the full frame image."""
    conn = get_db()
    try:
        row = conn.execute("SELECT file_path FROM frame WHERE id = ?", (frame_id,)).fetchone()
        if not row or not row["file_path"] or not Path(row["file_path"]).exists():
            return "Not found", 404
        return send_file(row["file_path"], mimetype="image/jpeg")
    finally:
        conn.close()


@app.route("/api/explorer/labels")
def api_explorer_labels():
    """Get all distinct labels for a series."""
    series_name = request.args.get("series", "").strip()
    if not series_name:
        return jsonify({"labels": []})
    conn = get_db()
    try:
        sr = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not sr:
            return jsonify({"labels": []})
        rows = conn.execute("""
            SELECT DISTINCT d.label
            FROM detection d
            JOIN frame f ON d.frame_id = f.id
            JOIN episode e ON f.episode_id = e.id
            JOIN season sn ON e.season_id = sn.id
            WHERE sn.series_id = ?
            ORDER BY d.label
        """, (sr["id"],)).fetchall()
        return jsonify({"labels": [r["label"] for r in rows]})
    finally:
        conn.close()


@app.route("/api/export")
def api_export():
    """Export all detections for a series as CSV or JSON."""
    series_name = request.args.get("series", "").strip()
    fmt = request.args.get("format", "csv").strip().lower()

    if not series_name:
        return jsonify({"error": "series is required"}), 400

    conn = get_db()
    try:
        sr = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not sr:
            return jsonify({"error": f"Series '{series_name}' not found"}), 404

        rows = conn.execute("""
            SELECT d.id, d.frame_id, d.label, d.detector, d.confidence,
                   d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, d.pixel_area,
                   sn.number as season_num, e.number as ep_num, e.title as episode_title,
                   f.timestamp, f.file_path as frame_path
            FROM detection d
            JOIN frame f ON d.frame_id = f.id
            JOIN episode e ON f.episode_id = e.id
            JOIN season sn ON e.season_id = sn.id
            WHERE sn.series_id = ?
            ORDER BY sn.number, e.number, f.timestamp, d.id
        """, (sr["id"],)).fetchall()

        data = []
        for r in rows:
            data.append({
                "id": r["id"],
                "frame_id": r["frame_id"],
                "label": r["label"],
                "detector": r["detector"],
                "confidence": round(r["confidence"], 4),
                "bbox_x": r["bbox_x"],
                "bbox_y": r["bbox_y"],
                "bbox_w": r["bbox_w"],
                "bbox_h": r["bbox_h"],
                "pixel_area": r["pixel_area"],
                "episode_code": f"S{r['season_num']:02d}E{r['ep_num']:02d}",
                "episode_title": r["episode_title"] or "",
                "timestamp": round(r["timestamp"], 2),
            })

        if fmt == "json":
            return Response(
                json.dumps(data, indent=2),
                mimetype="application/json",
                headers={"Content-Disposition": f"attachment; filename={series_name}_detections.json"},
            )
        else:
            import csv
            import io
            output = io.StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            csv_str = output.getvalue()
            return Response(
                csv_str,
                mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename={series_name}_detections.csv"},
            )
    finally:
        conn.close()


# ── API: Theme Analysis (new metrics) ─────────────────────────────────────────


@app.route("/api/analysis")
def api_analysis():
    """Theme-based analysis with multiple metric types.

    Metrics returned per theme per episode:
    - frame_pct: % of frames containing ANY label from theme (no double counting)
    - pixel_area_pct: sum(pixel_area) / (num_frames * width * height) * 100  (episode share)
    - pixel_minutes: sum(pixel_area * delta_t) / 60
    - label_breakdown: per-label frame_pct and pixel_area_pct within the theme

    Params: series, themes (comma-sep theme names), metric (frame_pct|pixel_area_pct|pixel_minutes)
    """
    series_name = request.args.get("series", "").strip()
    theme_names = request.args.get("themes", "").strip()
    confidence = float(request.args.get("confidence", 0.25))

    if not series_name or not theme_names:
        return jsonify({"error": "series and themes are required"}), 400

    conn = get_db()
    try:
        sr = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not sr:
            return jsonify({"error": f"Series '{series_name}' not found"}), 404
        series_id = sr["id"]

        # Get episodes
        episodes = conn.execute("""
            SELECT e.id, e.number as ep_num, s.number as season_num, e.title,
                   e.num_frames, e.width, e.height, e.duration_secs
            FROM episode e JOIN season s ON e.season_id = s.id
            WHERE s.series_id = ? ORDER BY s.number, e.number
        """, (series_id,)).fetchall()
        ep_codes = [f"S{e['season_num']:02d}E{e['ep_num']:02d}" for e in episodes]

        # Get theme labels
        themes_data = get_all_themes(conn)
        theme_map = {t["name"].lower(): t for t in themes_data}

        requested = [t.strip() for t in theme_names.split(",") if t.strip()]
        results = {"episodes": ep_codes, "themes": [], "confidence": confidence}

        for theme_name in requested:
            t = theme_map.get(theme_name.lower())
            if not t:
                continue

            labels = t["labels"]
            if not labels:
                continue

            placeholders = ",".join("?" * len(labels))
            theme_result = {
                "name": t["name"],
                "description": t.get("description", ""),
                "labels": labels,
                "episodes": [],
            }

            for ep in episodes:
                total_frames = max(ep["num_frames"], 1)
                frame_pixels = ep["width"] * ep["height"]
                total_budget = total_frames * frame_pixels

                # Frame presence: count DISTINCT frames containing ANY of these labels
                frames_with_theme = conn.execute(f"""
                    SELECT COUNT(DISTINCT f.id) as c
                    FROM detection d
                    JOIN frame f ON d.frame_id = f.id
                    WHERE f.episode_id = ? AND d.label IN ({placeholders})
                    AND d.confidence >= ?
                """, (ep["id"], *labels, confidence)).fetchone()["c"]

                frame_pct = (frames_with_theme / total_frames * 100) if total_frames > 0 else 0

                # Pixel area: sum all pixel_area for these labels
                pixel_data = conn.execute(f"""
                    SELECT COALESCE(SUM(d.pixel_area), 0) as total_area,
                           COALESCE(SUM(d.pixel_area * f.delta_t), 0) as pixel_time
                    FROM detection d
                    JOIN frame f ON d.frame_id = f.id
                    WHERE f.episode_id = ? AND d.label IN ({placeholders})
                    AND d.confidence >= ?
                """, (ep["id"], *labels, confidence)).fetchone()

                pixel_area_pct = (pixel_data["total_area"] / total_budget * 100) if total_budget > 0 else 0
                pixel_minutes = pixel_data["pixel_time"] / 60.0

                # Per-label breakdown
                label_breakdown = []
                for lbl in labels:
                    lbl_frames = conn.execute("""
                        SELECT COUNT(DISTINCT f.id) as c
                        FROM detection d JOIN frame f ON d.frame_id = f.id
                        WHERE f.episode_id = ? AND d.label = ? AND d.confidence >= ?
                    """, (ep["id"], lbl, confidence)).fetchone()["c"]

                    lbl_area = conn.execute("""
                        SELECT COALESCE(SUM(d.pixel_area), 0) as a
                        FROM detection d JOIN frame f ON d.frame_id = f.id
                        WHERE f.episode_id = ? AND d.label = ? AND d.confidence >= ?
                    """, (ep["id"], lbl, confidence)).fetchone()["a"]

                    label_breakdown.append({
                        "label": lbl,
                        "frame_pct": round(lbl_frames / total_frames * 100, 4) if total_frames > 0 else 0,
                        "pixel_area_pct": round(lbl_area / total_budget * 100, 4) if total_budget > 0 else 0,
                    })

                theme_result["episodes"].append({
                    "code": f"S{ep['season_num']:02d}E{ep['ep_num']:02d}",
                    "frame_pct": round(frame_pct, 4),
                    "pixel_area_pct": round(pixel_area_pct, 4),
                    "pixel_minutes": round(pixel_minutes, 4),
                    "label_breakdown": label_breakdown,
                })

            results["themes"].append(theme_result)

        return jsonify(results)
    finally:
        conn.close()


@app.route("/api/analysis/drilldown")
def api_analysis_drilldown():
    """Get individual detection instances for a theme+episode combination.
    Returns frame images and detection details for chart drill-down."""
    series_name = request.args.get("series", "").strip()
    episode_code = request.args.get("episode", "").strip()
    label = request.args.get("label", "").strip()
    theme_name = request.args.get("theme", "").strip()
    confidence = float(request.args.get("confidence", 0.25))
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(1, int(request.args.get("per_page", 20))))

    if not series_name or not episode_code:
        return jsonify({"error": "series and episode are required"}), 400

    ep_match = re.match(r"S(\d+)E(\d+)", episode_code, re.IGNORECASE)
    if not ep_match:
        return jsonify({"error": "Invalid episode format"}), 400

    conn = get_db()
    try:
        sr = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
        if not sr:
            return jsonify({"error": "Series not found"}), 404

        ep = conn.execute("""
            SELECT e.id FROM episode e
            JOIN season sn ON e.season_id = sn.id
            WHERE sn.series_id = ? AND sn.number = ? AND e.number = ?
        """, (sr["id"], int(ep_match.group(1)), int(ep_match.group(2)))).fetchone()
        if not ep:
            return jsonify({"error": "Episode not found"}), 404

        # Determine labels to query
        labels = []
        if label:
            labels = [label]
        elif theme_name:
            themes_data = get_all_themes(conn)
            for t in themes_data:
                if t["name"].lower() == theme_name.lower():
                    labels = t["labels"]
                    break

        if not labels:
            return jsonify({"detections": [], "total": 0})

        placeholders = ",".join("?" * len(labels))
        total = conn.execute(f"""
            SELECT COUNT(*) as c FROM detection d
            JOIN frame f ON d.frame_id = f.id
            WHERE f.episode_id = ? AND d.label IN ({placeholders}) AND d.confidence >= ?
        """, (ep["id"], *labels, confidence)).fetchone()["c"]

        offset = (page - 1) * per_page
        rows = conn.execute(f"""
            SELECT d.id, d.frame_id, d.label, d.detector, d.confidence,
                   d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, d.pixel_area,
                   f.timestamp, f.file_path
            FROM detection d
            JOIN frame f ON d.frame_id = f.id
            WHERE f.episode_id = ? AND d.label IN ({placeholders}) AND d.confidence >= ?
            ORDER BY f.timestamp, d.label
            LIMIT ? OFFSET ?
        """, (ep["id"], *labels, confidence, per_page, offset)).fetchall()

        detections = [{
            "id": r["id"], "frame_id": r["frame_id"], "label": r["label"],
            "detector": r["detector"], "confidence": round(r["confidence"], 4),
            "bbox_x": r["bbox_x"], "bbox_y": r["bbox_y"],
            "bbox_w": r["bbox_w"], "bbox_h": r["bbox_h"],
            "pixel_area": r["pixel_area"],
            "timestamp": round(r["timestamp"], 2),
            "has_frame": bool(r["file_path"]),
        } for r in rows]

        return jsonify({
            "detections": detections,
            "total": total,
            "page": page,
            "pages": max(1, (total + per_page - 1) // per_page),
        })
    finally:
        conn.close()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    """Run the Vaction web UI."""
    import argparse
    parser = argparse.ArgumentParser(description="Vaction Web UI")
    parser.add_argument("--port", type=int, default=5555, help="Port (default: 5555)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data-dir", default=None, help="Data directory")
    args = parser.parse_args()

    global _config
    overrides = {}
    if args.data_dir:
        overrides["data_dir"] = Path(args.data_dir)
    _config = get_config(**overrides)
    _config.ensure_dirs()

    print(f"\n  Vaction Web UI")
    print(f"  http://{args.host}:{args.port}\n")
    print(f"  Data directory: {_config.data_dir}")
    print(f"  Database: {_config.db_path}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
