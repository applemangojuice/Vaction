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

from flask import Flask, Response, jsonify, render_template, request

from vaction.config import VactionConfig, get_config
from vaction.db import (
    db_session, get_aliases, get_connection, init_db, insert_alias,
    create_theme, update_theme, delete_theme, get_all_themes, get_theme_labels,
    get_clip_prompts, add_clip_prompt, delete_clip_prompt, toggle_clip_prompt,
    seed_default_clip_prompts, load_vocabulary_preset, seed_default_themes,
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


@app.route("/add-videos")
def add_videos_page():
    return render_template("add_videos.html")


@app.route("/detect")
def detect_page():
    return render_template("detect.html")


@app.route("/search")
def search_page():
    return render_template("search.html")


@app.route("/status")
def status_page():
    return render_template("status.html")


@app.route("/vocabulary")
def vocabulary_page():
    return render_template("vocabulary.html")


@app.route("/themes")
def themes_page():
    return render_template("themes.html")


@app.route("/charts")
def charts_page():
    return render_template("charts.html")


@app.route("/faces")
def faces_page():
    return render_template("faces.html")


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
    batch_size = int(data.get("batch_size", 16))
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

                # If force mode, clear existing detections for this series first
                if force:
                    _job_put(job_id, "status", "Force mode: clearing existing detections...")
                    # Get the specific detectors being re-run
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
                    conn.commit()

                # Count total frames for this series
                total_all = conn.execute("""
                    SELECT COUNT(*) as c FROM frame f
                    JOIN episode e ON f.episode_id = e.id
                    JOIN season s ON e.season_id = s.id
                    JOIN series sr ON s.series_id = sr.id
                    WHERE sr.name = ?
                """, (series_name,)).fetchone()["c"]

                # Count frames with file paths
                total_with_files = conn.execute("""
                    SELECT COUNT(*) as c FROM frame f
                    JOIN episode e ON f.episode_id = e.id
                    JOIN season s ON e.season_id = s.id
                    JOIN series sr ON s.series_id = sr.id
                    WHERE sr.name = ? AND f.file_path IS NOT NULL
                """, (series_name,)).fetchone()["c"]

                # Count already-detected frames
                already_detected = conn.execute("""
                    SELECT COUNT(DISTINCT f.id) as c FROM frame f
                    JOIN detection d ON d.frame_id = f.id
                    JOIN episode e ON f.episode_id = e.id
                    JOIN season s ON e.season_id = s.id
                    JOIN series sr ON s.series_id = sr.id
                    WHERE sr.name = ?
                """, (series_name,)).fetchone()["c"]

                _job_put(job_id, "status", f"Series '{series_name}': {total_all} total frames, {total_with_files} with files, {already_detected} already detected")

                frames_query = """
                    SELECT f.id, f.file_path, e.width, e.height
                    FROM frame f
                    JOIN episode e ON f.episode_id = e.id
                    JOIN season s ON e.season_id = s.id
                    JOIN series sr ON s.series_id = sr.id
                    WHERE sr.name = ? AND f.file_path IS NOT NULL
                """
                params = [series_name]

                if not force:
                    frames_query += " AND f.id NOT IN (SELECT DISTINCT frame_id FROM detection)"

                frames_query += " ORDER BY f.id"
                frames = conn.execute(frames_query, params).fetchall()

                if not frames:
                    msg = "No unprocessed frames found."
                    if already_detected > 0 and not force:
                        msg += f" {already_detected} frames already detected. Enable 'Force re-detect' to re-run."
                    elif total_with_files == 0:
                        msg += f" {total_all} frames exist but none have file paths. Frame files may have been deleted."
                    _job_put(job_id, "status", msg)
                    _job_put(job_id, "complete", json.dumps({"total_detections": 0, "frames_processed": 0}))
                    return

                total_frames = len(frames)
                _job_put(job_id, "status", f"Processing {total_frames} frames...")
                total_detections = 0

                for i in range(0, total_frames, batch_size):
                    batch = frames[i:i + batch_size]
                    batch_dicts = [{"id": f["id"], "file_path": f["file_path"]} for f in batch]
                    width = batch[0]["width"]
                    height = batch[0]["height"]

                    count = registry.run_on_batch(conn, batch_dicts, width, height)
                    total_detections += count
                    conn.commit()

                    processed = min(i + batch_size, total_frames)
                    _job_put(job_id, "progress", json.dumps({
                        "processed": processed,
                        "total": total_frames,
                        "detections": total_detections,
                        "pct": round(processed / total_frames * 100, 1),
                    }))

                _job_put(job_id, "complete", json.dumps({
                    "total_detections": total_detections,
                    "frames_processed": total_frames,
                }))
            finally:
                conn.close()
        except Exception as e:
            import traceback
            _job_put(job_id, "error", f"{e}\n{traceback.format_exc()}")
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

    job_id = f"facescan_{int(time.time())}"
    _job_queues[job_id] = queue.Queue()
    _job_logs[job_id] = []
    _jobs[job_id] = {"type": "facescan", "status": "running", "series": series_name, "progress": None, "last_status": "", "page": "/faces"}

    def run_facescan():
        try:
            import numpy as np
            config = get_app_config()
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

                embeddings = []
                det_ids = []
                det_infos = []

                for i, det in enumerate(face_dets):
                    if not det["file_path"] or det["bbox_x"] is None:
                        continue
                    try:
                        import cv2
                        img = cv2.imread(det["file_path"])
                        if img is None:
                            continue
                        faces = detector._app.get(img)
                        # Find the face closest to our bbox
                        best_face = None
                        best_overlap = 0
                        for face in faces:
                            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                            # Simple overlap check
                            ox = max(0, min(fx2, det["bbox_x"] + det["bbox_w"]) - max(fx1, det["bbox_x"]))
                            oy = max(0, min(fy2, det["bbox_y"] + det["bbox_h"]) - max(fy1, det["bbox_y"]))
                            overlap = ox * oy
                            if overlap > best_overlap and face.embedding is not None:
                                best_overlap = overlap
                                best_face = face

                        if best_face is not None and best_face.embedding is not None:
                            embeddings.append(best_face.embedding.astype(np.float32))
                            det_ids.append(det["det_id"])
                            det_infos.append(det)
                    except Exception:
                        continue

                    if (i + 1) % 50 == 0:
                        _job_put(job_id, "progress", json.dumps({"processed": i + 1, "total": len(face_dets), "pct": round((i+1)/len(face_dets)*100, 1)}))

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
        except Exception as e:
            import traceback
            _job_put(job_id, "error", f"{e}\n{traceback.format_exc()}")
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
