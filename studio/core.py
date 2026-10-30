"""Paths, settings, ffmpeg helpers and the background job system."""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
BASE_DIR = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent.parent
RES_DIR = Path(getattr(sys, "_MEIPASS", BASE_DIR))
DATA_DIR = BASE_DIR / "StudioData"
BIN_DIR = BASE_DIR / "bin"
MODELS_DIR = DATA_DIR / "models"
SETTINGS_FILE = DATA_DIR / "settings.json"
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

DEFAULTS = {
    "output_dir": str(DATA_DIR / "Output"),
    "download_dir": str(DATA_DIR / "Downloads"),
    "openai_key": "",
    "pexels_key": "",
    "openai_chat_model": "gpt-4o-mini",
    "whisper_model": "medium",
    "encoder": "auto",
}

_settings_lock = threading.Lock()


def load_settings():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = dict(DEFAULTS)
    try:
        data.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return data


def save_settings(changes):
    with _settings_lock:
        data = load_settings()
        data.update({k: v for k, v in changes.items() if k in DEFAULTS})
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def tool(name):
    """Find a bundled binary first, then fall back to PATH."""
    exe = name + (".exe" if os.name == "nt" else "")
    bundled = BIN_DIR / exe
    if bundled.exists():
        return str(bundled)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} nahi mila. bin folder mein {exe} rakhein.")
    return found


def safe_name(text, fallback="output"):
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(text)).strip(" .")
    return text[:120] or fallback


def output_folder(kind, name):
    root = Path(load_settings()["output_dir"]) / kind
    folder = root / f"{safe_name(name)}_{time.strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def run(args, **kwargs):
    """Run a command hidden and raise with its stderr when it fails."""
    proc = subprocess.run(args, capture_output=True, creationflags=CREATE_NO_WINDOW, **kwargs)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-1500:]
        raise RuntimeError(f"{Path(str(args[0])).stem} fail hua:\n{tail}")
    return proc


def probe(path):
    out = run([tool("ffprobe"), "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", str(path)]).stdout
    return json.loads(out.decode("utf-8", "replace"))


def media_info(path):
    info = probe(path)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    result = {"duration": float(info["format"].get("duration") or 0), "has_audio": audio is not None,
              "has_video": video is not None}
    if video:
        num, den = (video.get("r_frame_rate") or "30/1").split("/")
        result.update(width=int(video["width"]), height=int(video["height"]),
                      fps=float(num) / max(float(den), 1))
    return result


def ffmpeg_progress(args, duration, job=None, start=0.0, end=1.0, stage=None, cwd=None):
    """Run ffmpeg, mapping its reported time into the job's progress range."""
    cmd = [tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostats",
           "-progress", "pipe:1"] + [str(a) for a in args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
                            creationflags=CREATE_NO_WINDOW, text=True, encoding="utf-8", errors="replace")
    errors = []
    reader = threading.Thread(target=lambda: errors.extend(proc.stderr), daemon=True)
    reader.start()
    try:
        for line in proc.stdout:
            if job:
                job.check()
            if line.startswith("out_time_us=") and job and duration:
                value = line.split("=", 1)[1].strip()
                if value.isdigit():
                    frac = min(int(value) / 1e6 / duration, 1.0)
                    job.update(start + (end - start) * frac, stage)
    except BaseException:
        proc.kill()
        raise
    proc.wait()
    reader.join(2)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg fail hua:\n" + "".join(errors)[-1500:])


_encoders = None


def video_encoder():
    """Pick the fastest working H.264 encoder and test it, since a listed encoder may lack a driver."""
    global _encoders
    choice = load_settings().get("encoder", "auto")
    if choice != "auto":
        return choice
    if _encoders is None:
        _encoders = "libx264"
        for enc in ("h264_nvenc", "h264_amf", "h264_qsv"):
            try:
                run([tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                     "color=c=black:s=320x240:d=0.2", "-c:v", enc, "-f", "null", "-"], timeout=20)
                _encoders = enc
                break
            except (RuntimeError, subprocess.TimeoutExpired):
                continue
    return _encoders


def encoder_args(quality=20):
    enc = video_encoder()
    if enc == "h264_nvenc":
        return ["-c:v", enc, "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0", "-pix_fmt", "yuv420p"]
    if enc == "h264_amf":
        return ["-c:v", enc, "-quality", "quality", "-rc", "cqp", "-qp_i", str(quality),
                "-qp_p", str(quality), "-pix_fmt", "yuv420p"]
    if enc == "h264_qsv":
        return ["-c:v", enc, "-global_quality", str(quality), "-pix_fmt", "nv12"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", str(quality), "-pix_fmt", "yuv420p"]


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, kind, title):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.title = title
        self.progress = 0.0
        self.stage = "Queue mein hai"
        self.status = "queued"
        self.result = None
        self.error = None
        self.started = time.time()
        self.finished = None
        self.cancel_flag = threading.Event()

    def update(self, progress=None, stage=None):
        self.check()
        if progress is not None:
            self.progress = max(self.progress, min(float(progress), 1.0))
        if stage:
            self.stage = stage

    def check(self):
        if self.cancel_flag.is_set():
            raise Cancelled()

    def as_dict(self):
        return {"id": self.id, "kind": self.kind, "title": self.title, "progress": round(self.progress, 4),
                "stage": self.stage, "status": self.status, "result": self.result, "error": self.error,
                "elapsed": round((self.finished or time.time()) - self.started, 1)}


class JobManager:
    def __init__(self, workers=2):
        self.jobs = {}
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, kind, title, func, *args, **kwargs):
        job = Job(kind, title)
        self.jobs[job.id] = job
        self.pool.submit(self._run, job, func, args, kwargs)
        return job

    def _run(self, job, func, args, kwargs):
        job.status = "running"
        job.started = time.time()
        try:
            job.result = func(job, *args, **kwargs)
            job.progress = 1.0
            job.stage = "Ho gaya"
            job.status = "done"
        except Cancelled:
            job.status = "cancelled"
            job.stage = "Cancel kiya gaya"
        except Exception as exc:
            traceback.print_exc()
            job.status = "error"
            job.error = str(exc)
            job.stage = "Error"
        job.finished = time.time()

    def cancel(self, job_id):
        job = self.jobs.get(job_id)
        if job:
            job.cancel_flag.set()

    def listing(self):
        return [j.as_dict() for j in sorted(self.jobs.values(), key=lambda j: j.started, reverse=True)][:40]


jobs = JobManager()
