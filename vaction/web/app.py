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
from vaction.db import db_session, get_aliases, get_connection, init_db, insert_alias

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

# Global state for background jobs
_jobs: dict[str, dict] = {}
_job_queues: dict[str, queue.Queue] = {}
_config: VactionConfig | None = None


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
    _jobs[job_id] = {"type": "ingest", "status": "running", "series": series_name}

    def run_ingest():
        q = _job_queues[job_id]
        try:
            config = get_app_config()
            config_overrides = {"default_fps": fps, "data_dir": config.data_dir}
            if transcode:
                config_overrides["transcode_resolution"] = int(transcode)
            cfg = get_config(**config_overrides)
            cfg.ensure_dirs()

            q.put({"event": "status", "data": "Scanning for video files..."})

            from vaction.ingest import ingest_directory
            conn = init_db(cfg.db_path)
            try:
                episode_ids = ingest_directory(conn, root, series_name, cfg)
                q.put({"event": "status", "data": f"Found {len(episode_ids)} episodes. Extracting frames at {fps} fps..."})

                if episode_ids:
                    from vaction.sampler import sample_episodes

                    def progress_cb(ep_id, count):
                        row = conn.execute("SELECT number FROM episode WHERE id = ?", (ep_id,)).fetchone()
                        ep_num = row["number"] if row else ep_id
                        q.put({"event": "progress", "data": json.dumps({
                            "episode": ep_num, "frames": count,
                            "done": False,
                        })})

                    results = sample_episodes(conn, episode_ids, cfg.frames_dir, fps=fps, progress_callback=progress_cb)
                    total_frames = sum(results.values())
                    q.put({"event": "complete", "data": json.dumps({
                        "episodes": len(episode_ids),
                        "total_frames": total_frames,
                    })})
                else:
                    q.put({"event": "complete", "data": json.dumps({"episodes": 0, "total_frames": 0})})
            finally:
                conn.close()
        except Exception as e:
            q.put({"event": "error", "data": str(e)})
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

    if not series_name:
        return jsonify({"error": "series is required"}), 400

    job_id = f"detect_{int(time.time())}"
    _job_queues[job_id] = queue.Queue()
    _jobs[job_id] = {"type": "detect", "status": "running", "series": series_name}

    def run_detect():
        q = _job_queues[job_id]
        try:
            config = get_app_config()
            conn = init_db(config.db_path)
            try:
                row = conn.execute("SELECT id FROM series WHERE name = ?", (series_name,)).fetchone()
                if not row:
                    q.put({"event": "error", "data": f"Series '{series_name}' not found"})
                    return

                from vaction.detectors.registry import DetectorRegistry
                registry = DetectorRegistry()
                detector_names = [d.strip() for d in detectors.split(",")]
                dev = config.device

                q.put({"event": "status", "data": f"Loading models: {', '.join(detector_names)}..."})

                for name in detector_names:
                    if name == "yolo":
                        from vaction.detectors.yolo import YOLODetector
                        registry.register(YOLODetector(model_name=config.yolo_model, device=dev, confidence=config.confidence_threshold))
                    elif name == "clip":
                        from vaction.detectors.clip import CLIPDetector
                        registry.register(CLIPDetector(model_name=config.clip_model, pretrained=config.clip_pretrained, device=dev))
                    elif name == "face":
                        from vaction.detectors.faces import FaceDetector
                        det = FaceDetector(device=dev, distance_threshold=config.face_distance_threshold)
                        det.load_character_embeddings(conn, row["id"])
                        registry.register(det)

                q.put({"event": "status", "data": "Models loaded. Querying frames..."})

                frames_query = """
                    SELECT f.id, f.file_path, e.width, e.height
                    FROM frame f
                    JOIN episode e ON f.episode_id = e.id
                    JOIN season s ON e.season_id = s.id
                    JOIN series sr ON s.series_id = sr.id
                    WHERE sr.name = ? AND f.file_path IS NOT NULL
                      AND f.id NOT IN (SELECT DISTINCT frame_id FROM detection)
                    ORDER BY f.id
                """
                frames = conn.execute(frames_query, (series_name,)).fetchall()

                if not frames:
                    q.put({"event": "complete", "data": json.dumps({"total_detections": 0, "frames_processed": 0})})
                    return

                total_frames = len(frames)
                q.put({"event": "status", "data": f"Processing {total_frames} frames..."})
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
                    q.put({"event": "progress", "data": json.dumps({
                        "processed": processed,
                        "total": total_frames,
                        "detections": total_detections,
                        "pct": round(processed / total_frames * 100, 1),
                    })})

                q.put({"event": "complete", "data": json.dumps({
                    "total_detections": total_detections,
                    "frames_processed": total_frames,
                })})
            finally:
                conn.close()
        except Exception as e:
            import traceback
            q.put({"event": "error", "data": f"{e}\n{traceback.format_exc()}"})
        finally:
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
        results = compute_episode_metrics(conn, query_str, confidence, series_id=row["id"])

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
    """Return data suitable for charting: episode share across episodes for one or more terms."""
    series_name = request.args.get("series", "").strip()
    terms = request.args.get("terms", "").strip()  # comma-separated
    confidence = float(request.args.get("confidence", 0.25))

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
            SELECT e.number as ep_num, s.number as season_num, e.title
            FROM episode e
            JOIN season s ON e.season_id = s.id
            WHERE s.series_id = ?
            ORDER BY s.number, e.number
        """, (row["id"],)).fetchall()

        ep_labels = [f"S{e['season_num']:02d}E{e['ep_num']:02d}" for e in episodes]

        term_list = [t.strip() for t in terms.split(",") if t.strip()]
        datasets = []

        for term in term_list:
            results = compute_episode_metrics(conn, term, confidence, series_id=row["id"])
            # Build a map of episode -> share
            ep_map = {}
            for r in results:
                key = f"S{r.season_number:02d}E{r.episode_number:02d}"
                ep_map[key] = round(r.episode_share_pct, 4)

            data_points = [ep_map.get(label, 0) for label in ep_labels]
            datasets.append({"term": term, "data": data_points})

        return jsonify({"labels": ep_labels, "datasets": datasets})
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
