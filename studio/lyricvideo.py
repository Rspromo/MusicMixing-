"""Lyric video builder: finds backgrounds, writes styled subtitles, renders with ffmpeg."""
import hashlib
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .core import (DATA_DIR, encoder_args, ffmpeg_progress, load_settings, media_info, output_folder, run, safe_name,
                   tool)
from .lyrics import save_text

CACHE = DATA_DIR / "media_cache"
UA = {"User-Agent": "StudioAutopilot/1.0 (local studio tool)"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
SIZES = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1)}


def pexels_search(term, key, per=6, vertical=False):
    found = []
    orient = "portrait" if vertical else "landscape"
    head = {"Authorization": key}
    videos = requests.get("https://api.pexels.com/videos/search", headers=head, timeout=20,
                          params={"query": term, "per_page": per, "orientation": orient})
    videos.raise_for_status()
    for item in videos.json().get("videos", []):
        files = [f for f in item["video_files"] if f.get("file_type") == "video/mp4" and (f.get("height") or 0) >= 720]
        if files:
            best = min(files, key=lambda f: f["height"])
            found.append({"url": best["link"], "kind": "video", "credit": f"Pexels: {item['user']['name']} ({item['url']})"})
    photos = requests.get("https://api.pexels.com/v1/search", headers=head, timeout=20,
                          params={"query": term, "per_page": per, "orientation": orient})
    photos.raise_for_status()
    for item in photos.json().get("photos", []):
        found.append({"url": item["src"]["large2x"], "kind": "image",
                      "credit": f"Pexels: {item['photographer']} ({item['url']})"})
    return found


def commons_search(term, per=8):
    data = requests.get("https://commons.wikimedia.org/w/api.php", headers=UA, timeout=20, params={
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrlimit": per * 2,
        "gsrsearch": f"filetype:bitmap {term}", "prop": "imageinfo", "iiprop": "url|size|extmetadata",
        "iiurlwidth": 1920}).json()
    found = []
    for page in data.get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        if info.get("width", 0) < 1280 or "fair use" in license_name.lower():
            continue
        artist = meta.get("Artist", {}).get("value", "unknown")
        found.append({"url": info.get("thumburl") or info["url"], "kind": "image",
                      "credit": f"Wikimedia Commons: {page['title']} | {license_name} | {artist[:120]} | {info.get('descriptionurl', '')}"})
    return found[:per]


def fetch(item):
    CACHE.mkdir(parents=True, exist_ok=True)
    ext = Path(item["url"].split("?")[0]).suffix or (".mp4" if item["kind"] == "video" else ".jpg")
    target = CACHE / (hashlib.sha1(item["url"].encode()).hexdigest()[:16] + ext)
    if target.exists() and target.stat().st_size:
        return target
    part = target.with_suffix(target.suffix + ".part")
    with requests.get(item["url"], headers=UA, stream=True, timeout=40) as reply:
        reply.raise_for_status()
        with part.open("wb") as handle:
            for chunk in reply.iter_content(1 << 16):
                handle.write(chunk)
    part.replace(target)
    return target


def collect(job, terms, source, need, local, vertical):
    items, credits = [], []
    for path in local:
        kind = "video" if Path(path).suffix.lower() in VIDEO_EXT else "image"
        items.append({"path": Path(path), "kind": kind})
    if source == "local" or len(items) >= need:
        return items, credits
    key = load_settings()["pexels_key"]
    found = []
    for i, term in enumerate(terms):
        job.update(0.03 + 0.05 * i / max(len(terms), 1), f"Visuals dhoondh raha hai: {term}")
        if key and source in ("mix", "pexels"):
            try:
                found += pexels_search(term, key, vertical=vertical)
            except requests.RequestException as exc:
                job.stage = f"Pexels error: {exc}"
        if source in ("mix", "commons"):
            try:
                found += commons_search(term)
            except (requests.RequestException, ValueError, KeyError):
                pass
    random.shuffle(found)
    found = found[:max(need, 4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda f: _safe_fetch(f), found))
    for i, (item, path) in enumerate(zip(found, results)):
        job.update(0.08 + 0.1 * (i + 1) / max(len(found), 1), "Visuals download ho rahe hain")
        if path:
            items.append({"path": path, "kind": item["kind"]})
            credits.append(item["credit"])
    return items, credits


def _safe_fetch(item):
    try:
        return fetch(item)
    except (requests.RequestException, OSError):
        return None


def ass_color(hex_color, alpha=0):
    value = hex_color.lstrip("#")
    return f"&H{alpha:02X}{value[4:6]}{value[2:4]}{value[0:2]}&".upper()


def ass_time(t):
    t = max(t, 0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def escape(text):
    return text.replace("\\", " ").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(lines, cfg, width, height, duration):
    scale = height / 1080 if width >= height else width / 1080
    size = int(cfg["font_size"] * scale)
    align = {"bottom": 2, "center": 5, "top": 8}[cfg["position"]]
    margin = int(height * 0.09)
    primary = ass_color(cfg["text_color"])
    high = ass_color(cfg["highlight_color"])
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n")
    style = (f"Style: Main,{cfg['font']},{size},{primary},{primary if cfg['style'] != 'karaoke' else ass_color(cfg['text_color'], 0x70)},"
             f"{ass_color(cfg['outline_color'])},&H80000000,{-1 if cfg['bold'] else 0},0,0,0,100,100,0,0,1,"
             f"{cfg['outline'] * scale:.1f},{cfg['shadow'] * scale:.1f},{align},{int(width * .06)},{int(width * .06)},{margin},1\n")
    next_style = (f"Style: Next,{cfg['font']},{int(size * .62)},{ass_color(cfg['text_color'], 0x80)},{primary},"
                  f"{ass_color(cfg['outline_color'], 0x80)},&H80000000,0,0,0,0,100,100,1,0,1,{cfg['outline'] * scale * .6:.1f},0,"
                  f"{align},{int(width * .06)},{int(width * .06)},{margin if align != 2 else max(margin - int(size * 1.0), 10)},1\n")
    title = (f"Style: Title,{cfg['font']},{int(size * 1.7)},{high},{primary},{ass_color(cfg['outline_color'])},&H80000000,"
             f"-1,0,0,0,100,100,0,0,1,{cfg['outline'] * scale:.1f},{cfg['shadow'] * scale:.1f},5,40,40,40,1\n")
    body = ("\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    events = []
    upper = cfg.get("uppercase")
    fade = int(cfg["fade_ms"])
    shift = float(cfg["offset"])
    first_line = lines[0]["start"] if lines else duration
    stop = min(first_line + shift - 0.2, 8)
    if cfg.get("title") and stop > 1.5:
        artist = f"\\N{{\\fs{int(size * .75)}\\c{primary}}}{escape(cfg['artist'])}" if cfg.get("artist") else ""
        events.append(f"Dialogue: 2,{ass_time(0.2)},{ass_time(stop)},Title,,0,0,0,,{{\\fad(600,600)}}{escape(cfg['title'])}{artist}")
    for i, line in enumerate(lines):
        start = line["start"] + shift
        follow = lines[i + 1]["start"] + shift if i + 1 < len(lines) else duration
        end = min(line["end"] + shift + 0.5, follow - 0.02) if follow > start + .2 else line["end"] + shift
        text = line["text"].upper() if upper else line["text"]
        words = line.get("words") or []
        tag = f"{{\\fad({fade},{fade})}}"
        kind = cfg["style"]
        if kind == "karaoke" and words:
            parts = []
            for k, w in enumerate(words):
                stop = words[k + 1]["start"] if k + 1 < len(words) else w["end"]
                cs = max(int((stop - w["start"]) * 100), 1)
                word = w["word"].upper() if upper else w["word"]
                parts.append(f"{{\\kf{cs}}}{escape(word)}")
            lead = max(int((words[0]["start"] - line["start"]) * 100), 0)
            text_out = f"{tag}{{\\c{high}\\k{lead}}}" + " ".join(parts)
        elif kind == "word_pop" and words:
            parts = []
            for w in words:
                ms = max(int((w["start"] - line["start"]) * 1000), 0)
                word = w["word"].upper() if upper else w["word"]
                parts.append(f"{{\\alpha&HFF&\\fscx70\\fscy70\\t({ms},{ms + 110},\\alpha&H00&\\fscx100\\fscy100)}}{escape(word)}")
            text_out = "{\\fad(0,%d)}" % fade + " ".join(parts)
        elif kind == "slide":
            cx = width // 2
            base = {2: height - margin, 5: height // 2, 8: margin}[align]
            text_out = f"{{\\fad({fade},{fade})\\move({cx},{base + 40},{cx},{base},0,{max(fade, 200)})}}{escape(text)}"
        else:
            text_out = tag + escape(text)
        events.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Main,,0,0,0,,{text_out}")
        if kind == "next_line" and i + 1 < len(lines):
            nxt = lines[i + 1]["text"]
            events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Next,,0,0,0,,{tag}{escape(nxt.upper() if upper else nxt)}")
    return head + style + next_style + title + body + "\n".join(events) + "\n"


def render_segment(args):
    item, index, length, width, height, fps, fade, kenburns, folder = args
    target = Path(folder) / f"seg_{index:04d}.mp4"
    scale = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1"
    if item["kind"] == "image":
        zoom_w, zoom_h = int(width * 1.15) // 2 * 2, int(height * 1.15) // 2 * 2
        pre = f"scale={zoom_w}:{zoom_h}:force_original_aspect_ratio=increase,crop={zoom_w}:{zoom_h}"
        if kenburns:
            direction = index % 4
            x = ["(iw-ow)*t/{d}", "(iw-ow)*(1-t/{d})", "(iw-ow)/2", "(iw-ow)/2"][direction].format(d=length)
            y = ["(ih-oh)/2", "(ih-oh)/2", "(ih-oh)*t/{d}", "(ih-oh)*(1-t/{d})"][direction].format(d=length)
            vf = f"{pre},crop={width}:{height}:x='{x}':y='{y}',setsar=1"
        else:
            vf = scale
        inputs = ["-loop", "1", "-framerate", str(fps), "-t", str(length), "-i", str(item["path"])]
    else:
        vf = scale
        inputs = ["-stream_loop", "-1", "-t", str(length), "-i", str(item["path"])]
    vf += f",fps={fps},format=yuv420p,fade=in:st=0:d={fade},fade=out:st={max(length - fade, 0):.3f}:d={fade}"
    run([tool("ffmpeg"), "-y", "-v", "error"] + inputs + ["-vf", vf, "-an", "-c:v", "libx264", "-preset", "ultrafast",
                                                          "-crf", "17", str(target)])
    return target


DEFAULT = {"aspect": "16:9", "height": 1080, "fps": 30, "font": "Nirmala UI", "font_size": 74, "text_color": "#FFFFFF",
           "highlight_color": "#FFC94D", "outline_color": "#000000", "outline": 3, "shadow": 2, "bold": True,
           "position": "center", "style": "karaoke", "fade_ms": 220, "offset": 0.0, "uppercase": False,
           "source": "mix", "clip_length": 7, "kenburns": True, "blur": 0, "dim": 0.35, "crossfade": 0.6,
           "visualizer": False, "title": "", "artist": "", "logo": "", "quality": 20, "preview": 0}


def render(job, audio, lines, terms, local=None, options=None):
    cfg = {**DEFAULT, **(options or {})}
    lines = sorted(lines, key=lambda l: l["start"])
    name = safe_name(cfg["title"] or Path(audio).stem)
    out = output_folder("LyricVideos", name)
    temp = out / "_temp"
    temp.mkdir()
    total = media_info(audio)["duration"]
    duration = min(total, float(cfg["preview"])) if float(cfg["preview"]) > 0 else total
    w_ratio, h_ratio = SIZES[cfg["aspect"]]
    short = int(cfg["height"])
    if w_ratio >= h_ratio:
        height, width = short, int(short * w_ratio / h_ratio) // 2 * 2
    else:
        width, height = short, int(short * h_ratio / w_ratio) // 2 * 2
    length = float(cfg["clip_length"])
    count = max(math.ceil(duration / length), 1)
    terms = [t for t in terms if t.strip()] or [Path(audio).stem, "cinematic light", "abstract texture"]
    items, credits = collect(job, terms, cfg["source"], min(count, 40), local or [], w_ratio < h_ratio)
    if not items:
        raise RuntimeError("Koi background nahi mila. Pexels key daalein, keywords badlein ya apni images dein.")

    job.update(0.2, "Background clips bana raha hai")
    tasks = [(items[i % len(items)], i, length, width, height, cfg["fps"], cfg["crossfade"], cfg["kenburns"], str(temp))
             for i in range(count)]
    segments = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, seg in enumerate(pool.map(render_segment, tasks), start=1):
            job.update(0.2 + 0.3 * i / count, f"Background {i}/{count}")
            segments.append(seg)
    (temp / "list.txt").write_text("".join(f"file '{s.name}'\n" for s in segments), encoding="utf-8")
    (temp / "lyrics.ass").write_text(build_ass(lines, cfg, width, height, duration), encoding="utf-8-sig")

    chain = [f"[0:v]setpts=PTS-STARTPTS"]
    if float(cfg["blur"]) > 0:
        chain.append(f"gblur=sigma={float(cfg['blur'])}")
    if float(cfg["dim"]) > 0:
        chain.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=black@{float(cfg['dim']):.2f}:t=fill")
    graph = ",".join(chain) + "[bg]"
    last = "bg"
    if cfg["visualizer"]:
        graph += (f";[1:a]showwaves=s={width}x{height // 6}:mode=cline:rate={cfg['fps']}:colors="
                  f"{cfg['highlight_color'].replace('#', '0x')},format=rgba,colorchannelmixer=aa=0.65[wave]"
                  f";[{last}][wave]overlay=0:H-h[wv]")
        last = "wv"
    extra = []
    if cfg["logo"]:
        extra = ["-i", cfg["logo"]]
        graph += f";[2:v]scale={width // 9}:-1[logo];[{last}][logo]overlay=W-w-{width // 30}:{width // 30}[lg]"
        last = "lg"
    graph += f";[{last}]ass=lyrics.ass:shaping=complex[out]"  # complex shaping joins Devanagari matras
    target = out / f"{name}_LyricVideo.mp4"
    args = ["-f", "concat", "-safe", "0", "-i", "list.txt", "-i", str(Path(audio).resolve())] + extra + [
        "-filter_complex", graph, "-map", "[out]", "-map", "1:a:0", "-t", f"{duration:.3f}",
        *encoder_args(int(cfg["quality"])), "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", str(target.resolve())]
    ffmpeg_progress(args, duration, job, 0.5, 0.98, "Final video render ho raha hai", cwd=str(temp))
    save_text(lines, out, name)
    if credits:
        (out / "Image_Credits.txt").write_text("\n".join(sorted(set(credits))), encoding="utf-8")
    for leftover in temp.iterdir():
        leftover.unlink()
    temp.rmdir()
    return {"folder": str(out), "file": str(target), "credits": len(set(credits))}
