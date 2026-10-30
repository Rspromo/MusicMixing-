"""Link downloader with a local library that stores BPM and key."""
import html
import json
import re
import threading
import time
from pathlib import Path

import requests

from .analysis import analyze_file
from .core import BIN_DIR, DATA_DIR, load_settings, safe_name, tool

LIBRARY = DATA_DIR / "library.json"
_lock = threading.Lock()
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"


def read_library():
    try:
        items = json.loads(LIBRARY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    for item in items:
        item["exists"] = Path(item["path"]).exists()
    return items


def _save(items):
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    LIBRARY.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")


def _add(entry):
    with _lock:
        items = [i for i in read_library() if i["path"] != entry["path"]]
        items.insert(0, entry)
        _save(items)


def remove_entry(path):
    with _lock:
        _save([i for i in read_library() if i["path"] != path])


def spotify_to_search(url):
    """Spotify audio is DRM protected, so find the same track on YouTube from its public title."""
    if "/track/" not in url:
        raise RuntimeError("Spotify mein abhi sirf single track link chalta hai, playlist ya album nahi.")
    data = requests.get("https://open.spotify.com/oembed", params={"url": url}, timeout=20).json()
    embed = requests.get(data["iframe_url"], headers={"User-Agent": BROWSER_UA}, timeout=20)
    artist = re.search(r'"artists":\[\{"name":"(.+?)"', embed.text)
    if artist:
        name = html.unescape(artist.group(1))
        return f"{name} {data['title']} official audio", f"{name} - {data['title']}"
    return f"{data['title']} official audio", data["title"]


def _base_options(job, folder, start, end):
    def hook(status):
        job.check()
        if status["status"] == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            if total:
                speed = (status.get("speed") or 0) / 1e6
                job.update(start + (end - start) * status["downloaded_bytes"] / total,
                           f"Download ho raha hai ({speed:.1f} MB/s)")
        elif status["status"] == "finished":
            job.update(end, "Convert ho raha hai")

    options = {
        "outtmpl": str(Path(folder) / "%(title).150B.%(ext)s"),
        "windowsfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "ffmpeg_location": str(Path(tool("ffmpeg")).parent),
        "concurrent_fragment_downloads": 4,
    }
    deno = BIN_DIR / "deno.exe"
    if deno.exists():
        options["js_runtimes"] = {"deno": {"path": str(deno)}}
    else:
        options["js_runtimes"] = {"deno": {}, "node": {}}
    return options


def download(job, url, kind="mp3", quality="best"):
    import yt_dlp

    url = url.strip()
    folder = Path(load_settings()["download_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    query, shown = url, None
    if "open.spotify.com" in url:
        job.update(0.02, "Spotify se song ka naam le raha hai")
        query, shown = spotify_to_search(url)
        query = "ytsearch1:" + query
    options = _base_options(job, folder, 0.05, 0.75)
    if kind in ("mp3", "wav"):
        options["format"] = "bestaudio/best"
        pp = {"key": "FFmpegExtractAudio", "preferredcodec": kind}
        if kind == "mp3":
            pp["preferredquality"] = "320"
        options["postprocessors"] = [pp]
    else:
        limit = "" if quality == "best" else f"[height<={int(quality)}]"
        options["format"] = f"bestvideo{limit}+bestaudio/best{limit}/best"
        options["merge_output_format"] = "mp4"
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(query, download=True)
        if info.get("entries"):
            info = info["entries"][0]
        final = (info.get("requested_downloads") or [{}])[0].get("filepath") or ydl.prepare_filename(info)
    path = Path(final)
    if not path.exists():
        wanted = path.with_suffix("." + kind)
        if wanted.exists():
            path = wanted
    if not path.exists():
        raise RuntimeError("Download khatam hua par file nahi mili.")
    job.update(0.8, "BPM aur Key nikal raha hai")
    facts = analyze_file(path)
    entry = {"title": shown or info.get("title") or path.stem, "path": str(path), "kind": kind, "url": url,
             "bpm": facts["bpm"], "key": facts["key"], "duration": facts["duration"],
             "size": path.stat().st_size, "added": time.strftime("%Y-%m-%d %H:%M")}
    _add(entry)
    return {"folder": str(folder), "file": str(path), "bpm": facts["bpm"], "key": facts["key"]}


def add_local(job, paths):
    """Analyse files the user already has and list them in the library."""
    for i, name in enumerate(paths):
        path = Path(name)
        job.update(i / len(paths), f"Analyse: {path.name}")
        facts = analyze_file(path)
        _add({"title": path.stem, "path": str(path), "kind": path.suffix[1:].lower(), "url": "",
              "bpm": facts["bpm"], "key": facts["key"], "duration": facts["duration"],
              "size": path.stat().st_size, "added": time.strftime("%Y-%m-%d %H:%M")})
    return {"count": len(paths)}
