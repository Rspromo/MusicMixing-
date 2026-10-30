"""Local web server behind the desktop window."""
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from . import APP_NAME, VERSION
from .core import RES_DIR, jobs, load_settings, media_info, save_settings, tool, video_encoder

TOKEN = secrets.token_urlsafe(24)
app = Flask(__name__, static_folder=None)
last_seen = time.time()
pick_lock = threading.Lock()


@app.before_request
def guard():
    # Any website can reach localhost, so every API call must carry this run's token.
    if request.path.startswith("/api/") and request.headers.get("X-Token") != TOKEN:
        if request.args.get("token") != TOKEN:
            abort(403)


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    html = (RES_DIR / "ui" / "index.html").read_text(encoding="utf-8")
    return html.replace("__TOKEN__", TOKEN).replace("__APP__", APP_NAME).replace("__VERSION__", VERSION)


@app.get("/ui/<path:name>")
def assets(name):
    return send_from_directory(RES_DIR / "ui", name)


def idle_seconds():
    """Seconds since the page last polled, or zero while any work is still running."""
    if any(j.status in ("queued", "running") for j in jobs.jobs.values()):
        return 0
    return time.time() - last_seen


def body():
    return request.get_json(force=True, silent=True) or {}


def job_reply(job):
    return jsonify({"job": job.id})


@app.get("/api/status")
def status():
    checks = {}
    for name in ("ffmpeg", "ffprobe"):
        try:
            checks[name] = bool(tool(name))
        except RuntimeError:
            checks[name] = False
    return jsonify({"tools": checks, "encoder": video_encoder() if checks["ffmpeg"] else "none",
                    "cpu": os.cpu_count()})


@app.get("/api/jobs")
def job_list():
    global last_seen
    last_seen = time.time()
    return jsonify(jobs.listing())


@app.post("/api/jobs/<job_id>/cancel")
def job_cancel(job_id):
    jobs.cancel(job_id)
    return jsonify(ok=True)


@app.get("/api/settings")
def settings_get():
    return jsonify(load_settings())


@app.post("/api/settings")
def settings_set():
    return jsonify(save_settings(body()))


def _tk_pick(mode, types):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if mode == "folder":
        picked = filedialog.askdirectory(parent=root)
        result = [picked] if picked else []
    elif mode == "files":
        result = list(filedialog.askopenfilenames(parent=root, filetypes=types))
    else:
        picked = filedialog.askopenfilename(parent=root, filetypes=types)
        result = [picked] if picked else []
    root.destroy()
    return result


@app.post("/api/pick")
def pick():
    data = body()
    filters = {"audio": ("Audio", "*.wav *.aif *.aiff *.flac *.mp3 *.m4a *.ogg"),
               "video": ("Video", "*.mp4 *.mov *.mkv *.mxf *.avi *.m4v *.mts"),
               "media": ("Media", "*.jpg *.jpeg *.png *.webp *.mp4 *.mov *.mkv *.webm"),
               "als": ("Ableton Live Set", "*.als"), "image": ("Image", "*.png *.jpg *.jpeg")}
    chosen = filters.get(data.get("types"))
    types = ([chosen] if chosen else []) + [("All files", "*.*")]
    result = []

    def worker():
        with pick_lock:
            result.extend(_tk_pick(data.get("mode", "file"), types))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    return jsonify([path.replace("/", "\\") for path in result])


@app.post("/api/open")
def open_path():
    path = Path(body().get("path", ""))
    if not path.exists():
        return jsonify(error="Ye path ab maujood nahi hai."), 404
    if path.is_file():
        subprocess.Popen(["explorer", "/select,", str(path)])
    else:
        os.startfile(str(path))
    return jsonify(ok=True)


@app.get("/api/media")
def media():
    path = Path(request.args.get("path", ""))
    if not path.is_file():
        abort(404)
    return send_file(path, conditional=True)


# Mixing
@app.post("/api/mix/scan")
def mix_scan():
    from .mixer import GENRES, ROLES, scan
    folder = body().get("folder", "")
    if not Path(folder).is_dir():
        return jsonify(error="Folder nahi mila."), 400
    return jsonify({"stems": scan(folder), "roles": ROLES,
                    "genres": {k: v["label"] for k, v in GENRES.items()}})


@app.get("/api/mix/genres")
def mix_genres():
    from .mixer import GENRES, ROLES
    return jsonify({"roles": ROLES, "genres": {k: v["label"] for k, v in GENRES.items()}})


@app.post("/api/mix/run")
def mix_run():
    from .mixer import mix_song
    data = body()
    stems = data.get("stems", [])
    name = data.get("name") or "Song"
    return job_reply(jobs.submit("mix", f"Mix: {name}", mix_song, stems, name, data.get("genre", "bollywood"),
                                 data.get("reference") or None, data.get("lufs") or None,
                                 float(data.get("vocal_boost", 0)), float(data.get("reverb", 1)),
                                 bool(data.get("variants", True)), bool(data.get("export_stems", True))))


# Downloader
@app.post("/api/download")
def download_start():
    from .downloader import download
    data = body()
    url = data.get("url", "").strip()
    if not url.startswith("http"):
        return jsonify(error="Sahi link daalein."), 400
    return job_reply(jobs.submit("download", f"Download: {url[:50]}", download, url,
                                 data.get("kind", "mp3"), data.get("quality", "best")))


@app.get("/api/library")
def library():
    from .downloader import read_library
    return jsonify(read_library())


@app.post("/api/library/remove")
def library_remove():
    from .downloader import remove_entry
    remove_entry(body().get("path", ""))
    return jsonify(ok=True)


@app.post("/api/library/add")
def library_add():
    from .downloader import add_local
    return job_reply(jobs.submit("analyse", "BPM aur Key nikal raha hai", add_local, body().get("paths", [])))


# Lyrics and lyric video
@app.post("/api/lyrics/extract")
def lyrics_extract():
    from .lyrics import extract
    data = body()
    audio = data.get("vocal") or data.get("audio", "")
    if not Path(audio).is_file():
        return jsonify(error="Audio file nahi mili."), 400
    return job_reply(jobs.submit("lyrics", f"Lyrics: {Path(audio).name}", extract, audio, data.get("language", ""),
                                 data.get("engine", "local"), data.get("model"), bool(data.get("polish")),
                                 bool(data.get("roman")), Path(data.get("audio", audio)).stem))


@app.post("/api/lyrics/render")
def lyrics_render():
    from .lyricvideo import render
    data = body()
    audio = data.get("audio", "")
    if not Path(audio).is_file():
        return jsonify(error="Audio file nahi mili."), 400
    return job_reply(jobs.submit("lyricvideo", f"Lyric video: {Path(audio).name}", render, audio, data.get("lines", []),
                                 data.get("terms", []), data.get("local", []), data.get("options", {})))


# Video editor
@app.get("/api/video/styles")
def video_styles():
    from .videoeditor import STYLES
    return jsonify({k: {"label": v["label"], "about": v["about"]} for k, v in STYLES.items()})


@app.post("/api/video/scan")
def video_scan():
    from .videoeditor import VIDEO_EXT
    folder = Path(body().get("folder", ""))
    if not folder.is_dir():
        return jsonify(error="Folder nahi mila."), 400
    return jsonify(sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT))


@app.post("/api/video/edit")
def video_edit():
    from .videoeditor import edit
    data = body()
    song = data.get("song", "")
    if not Path(song).is_file():
        return jsonify(error="Song file nahi mili."), 400
    return job_reply(jobs.submit("video", f"Video edit: {Path(song).name}", edit, song, data.get("clips", []),
                                 data.get("style", "performance"), int(data.get("fps", 30)), int(data.get("seed", 1)),
                                 float(data.get("length", 0)), int(data.get("quality", 20))))


# Tuning
@app.get("/api/tune/options")
def tune_options():
    from .analysis import NOTES
    from .tuner import SCALES
    return jsonify({"notes": NOTES, "scales": list(SCALES)})


@app.post("/api/tune/run")
def tune_run():
    from .tuner import tune
    data = body()
    vocals = data.get("vocals", [])
    return job_reply(jobs.submit("tune", f"Tuning: {len(vocals)} vocal", tune, vocals, data.get("music") or None,
                                 data.get("key") or None, data.get("scale") or None, data.get("preset", "natural"),
                                 data.get("strength"), float(data.get("max_shift", 1.3))))


@app.post("/api/ableton/bounce")
def ableton_bounce():
    from .ableton import bounce
    path = body().get("path", "")
    if not Path(path).is_file():
        return jsonify(error="Project file nahi mili."), 400
    return job_reply(jobs.submit("ableton", f"Ableton: {Path(path).name}", bounce, path))


@app.post("/api/info")
def info():
    return jsonify(media_info(body().get("path", "")))
