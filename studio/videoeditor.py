"""Automatic music video editor with Premiere Pro export."""
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET
from xml.dom import minidom

import numpy as np
from scipy import signal

from .analysis import band_features, beat_track, decode_mono, estimate_bpm, onset_envelope, onset_from_bands
from .core import CREATE_NO_WINDOW, encoder_args, ffmpeg_progress, media_info, output_folder, run, safe_name, tool

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".mxf", ".avi", ".m4v", ".mts"}
SYNC_RATE = 22050 / 220
SYNC_MIN = 0.45

STYLES = {
    "cinematic": {"label": "Cinematic Slow", "low": 8, "high": 4, "transition": "dissolve", "grade": "teal_orange",
                  "punch": 1.0, "sync": 0.5, "bars": True, "about": "Lambe shots, dissolves, teal-orange film look, letterbox"},
    "beat": {"label": "Beat Sync Fast", "low": 4, "high": 2, "transition": "cut", "grade": "contrast",
             "punch": 1.08, "sync": 0.5, "bars": False, "about": "Har beat par cut, punch-in, high contrast"},
    "performance": {"label": "Performance / Lip-sync", "low": 8, "high": 4, "transition": "cut", "grade": "clean",
                    "punch": 1.0, "sync": 0.85, "bars": False, "about": "Singer ke lip-sync shots par focus, kam b-roll"},
    "romantic": {"label": "Romantic Soft", "low": 12, "high": 8, "transition": "dissolve", "grade": "warm",
                 "punch": 1.0, "sync": 0.4, "bars": False, "about": "Bahut slow cuts, warm soft colours, dissolves"},
    "punjabi": {"label": "Punjabi High Energy", "low": 4, "high": 2, "transition": "flash", "grade": "punchy",
                "punch": 1.12, "sync": 0.6, "bars": False, "about": "Fast cuts, white flash, bright punchy colours"},
    "bw": {"label": "Black & White Moody", "low": 8, "high": 4, "transition": "cut", "grade": "bw",
           "punch": 1.0, "sync": 0.6, "bars": True, "about": "Moody monochrome look"},
    "reel": {"label": "Instagram Reel 9:16", "low": 4, "high": 2, "transition": "flash", "grade": "punchy",
             "punch": 1.05, "sync": 0.6, "bars": False, "about": "Vertical format, fast, reels ke liye"},
}


def write_cube(path, grade, size=25):
    """Write the look as a .cube LUT, usable in ffmpeg and Premiere's Lumetri alike."""
    axis = np.linspace(0, 1, size)
    lines = [f'TITLE "{grade}"', f"LUT_3D_SIZE {size}"]
    for b in axis:
        for g in axis:
            for r in axis:
                color = np.array([r, g, b])
                luma = color @ [0.2126, 0.7152, 0.0722]
                if grade == "teal_orange":
                    color = color + (np.array([0.09, 0.02, -0.08]) * luma * (1 - luma) * 3
                                     + np.array([-0.05, 0.01, 0.07]) * (1 - luma) ** 2)
                    color = 0.5 + (color - 0.5) * 1.1
                    color = luma + (color - luma) * 0.9
                elif grade == "warm":
                    color = color * [1.05, 1.0, 0.92] + 0.02
                    color = 0.5 + (color - 0.5) * 0.95
                elif grade == "contrast":
                    color = 0.5 + (color - 0.5) * 1.18
                elif grade == "punchy":
                    color = 0.5 + (color - 0.5) * 1.15
                    color = luma + (color - luma) * 1.25
                elif grade == "bw":
                    gray = 0.5 + (luma - 0.5) * 1.25
                    color = np.array([gray, gray, gray])
                elif grade == "clean":
                    color = 0.5 + (color - 0.5) * 1.06
                color = np.clip(color, 0, 1)
                lines.append(f"{color[0]:.5f} {color[1]:.5f} {color[2]:.5f}")
    Path(path).write_text("\n".join(lines), encoding="ascii")


def find_sync(song_bands, clip_path):
    """Align the camera's scratch audio to the song.

    Onsets give candidates, but a looped beat lines up on many bars, so the full
    spectrum then decides which candidate really matches the singing.
    """
    try:
        y = decode_mono(clip_path, 22050)
    except RuntimeError:
        return None, 0
    if len(y) < 22050 * 4 or np.abs(y).max() < 1e-3:
        return None, 0
    bands = band_features(y, 22050, hop=220)
    song_env, env = onset_from_bands(song_bands), onset_from_bands(bands)
    corr = signal.fftconvolve(song_env - song_env.mean(), (env - env.mean())[::-1], mode="full")
    peaks, _ = signal.find_peaks(corr, distance=15)
    top = peaks[np.argsort(corr[peaks])[::-1][:12]]
    clip = bands - bands.mean(axis=0)
    best = (0.0, None)
    for peak in top:
        lag = int(peak) - (len(env) - 1)
        first, last = max(lag, 0), min(lag + len(bands), len(song_bands))
        if last - first < SYNC_RATE * 3:
            continue
        a = song_bands[first:last]
        b = clip[first - lag:last - lag]
        a = a - a.mean(axis=0)
        score = float((a * b).sum() / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-9))
        if score > best[0]:
            best = (score, lag / SYNC_RATE)
    return best[1], best[0]


def score_clip(path, info):
    """Score each second for exposure, sharpness and movement from tiny grey frames."""
    w, h = 96, 54
    proc = subprocess.run([tool("ffmpeg"), "-v", "error", "-i", str(path), "-vf", f"fps=2,scale={w}:{h},format=gray",
                           "-f", "rawvideo", "-"], capture_output=True, creationflags=CREATE_NO_WINDOW)
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    count = len(raw) // (w * h)
    if count == 0:
        return np.ones(max(int(info["duration"] * 2), 1))
    frames = raw[:count * w * h].reshape(count, h, w).astype(np.float32)
    bright = frames.mean(axis=(1, 2))
    sharp = np.abs(np.diff(frames, axis=1)).mean(axis=(1, 2)) + np.abs(np.diff(frames, axis=2)).mean(axis=(1, 2))
    motion = np.abs(np.diff(frames, axis=0, prepend=frames[:1])).mean(axis=(1, 2))
    exposure = 1 - np.clip(np.abs(bright - 115) / 115, 0, 1)
    score = exposure * 0.4 + np.clip(sharp / (np.percentile(sharp, 90) + 1e-6), 0, 1) * 0.4 + np.clip(motion / 12, 0, 1) * 0.2
    score[motion > 50] *= 0.3  # flash frames and whip pans make weak shots
    return score


def analyse_clip(args):
    path, song_bands = args
    info = media_info(path)
    info["path"] = str(path)
    info["name"] = Path(path).name
    info["score"] = score_clip(path, info)
    offset, confidence = find_sync(song_bands, path) if info["has_audio"] else (None, 0)
    info["sync"] = offset if confidence > SYNC_MIN else None
    info["sync_confidence"] = round(confidence, 2)
    info["used"] = []
    return info


def build_cuts(beats, energy, style, length):
    cuts = []
    i = 0
    while i < len(beats) - 1 and beats[i] < length:
        hot = energy[min(i, len(energy) - 1)] > 0.6
        step = style["high"] if hot else style["low"]
        step = max(step, 1)
        j = min(i + step, len(beats) - 1)
        cuts.append((beats[i], beats[j]))
        i = j
    if cuts and cuts[-1][1] < length - 0.05:
        cuts.append((cuts[-1][1], length))
    return cuts


def pick(clips, start, end, style, last, rng):
    span = end - start
    synced = [c for c in clips if c["sync"] is not None and c["sync"] <= start and end - c["sync"] <= c["duration"] - 0.05]
    use_sync = synced and rng.random() < style["sync"]
    if use_sync:
        options = [c for c in synced if c["name"] != last] or synced
        def rate(c):
            at = int((start - c["sync"]) * 2)
            return float(np.mean(c["score"][at:at + max(int(span * 2), 1)]) if at < len(c["score"]) else 0) + rng.random() * .2
        clip = max(options, key=rate)
        return clip, start - clip["sync"]
    broll = [c for c in clips if c["duration"] >= span and c["name"] != last] or [c for c in clips if c["duration"] >= span]
    if not broll:
        # Nothing is long enough, so reuse the longest clip for as long as it lasts.
        clip = max(clips, key=lambda c: c["duration"])
        return clip, 0.0
    best = None
    for clip in broll:
        steps = max(int((clip["duration"] - span) * 2), 1)
        size = max(int(span * 2), 1)
        scores = np.convolve(clip["score"], np.ones(size) / size, mode="valid") if len(clip["score"]) >= size else clip["score"][:1]
        for at in range(0, min(steps, len(scores)), max(steps // 12, 1)):
            t = at / 2
            overlap = sum(1 for a, b in clip["used"] if t < b and t + span > a)
            value = float(scores[at]) - overlap * 0.6 - len(clip["used"]) * 0.03 + rng.random() * .1
            if best is None or value > best[0]:
                best = (value, clip, t)
    return best[1], best[2]


def seconds_url(path):
    resolved = Path(path).resolve().as_posix()
    if len(resolved) > 1 and resolved[1] == ":":
        resolved = resolved[0] + "%3a" + resolved[2:]
    return "file://localhost/" + quote(resolved, safe="/%")


def _node(parent, tag, text=None, **attrs):
    item = ET.SubElement(parent, tag, attrs)
    if text is not None:
        item.text = str(text)
    return item


def _rate(parent, fps):
    rate = _node(parent, "rate")
    _node(rate, "timebase", fps)
    _node(rate, "ntsc", "FALSE")


def _file_node(parent, file_id, path, info, fps, seen):
    if file_id in seen:
        _node(parent, "file", id=file_id)
        return
    seen.add(file_id)
    node = _node(parent, "file", id=file_id)
    _node(node, "name", Path(path).name)
    _node(node, "pathurl", seconds_url(path))
    _rate(node, fps)
    _node(node, "duration", int(info["duration"] * fps))
    media = _node(node, "media")
    if info.get("has_video"):
        video = _node(media, "video")
        char = _node(video, "samplecharacteristics")
        _node(char, "width", info["width"])
        _node(char, "height", info["height"])
    if info.get("has_audio"):
        audio = _node(media, "audio")
        char = _node(audio, "samplecharacteristics")
        _node(char, "depth", 16)
        _node(char, "samplerate", 48000)


def write_premiere(path, name, timeline, song, song_info, fps, width, height, punch, transition):
    root = ET.Element("xmeml", version="4")
    sequence = _node(root, "sequence", id="sequence-1")
    _node(sequence, "name", name)
    total = max(e["frame_end"] for e in timeline)
    _node(sequence, "duration", total)
    _rate(sequence, fps)
    code = _node(sequence, "timecode")
    _rate(code, fps)
    _node(code, "string", "00:00:00:00")
    _node(code, "frame", 0)
    _node(code, "displayformat", "NDF")
    media = _node(sequence, "media")
    video = _node(media, "video")
    fmt = _node(_node(video, "format"), "samplecharacteristics")
    _rate(fmt, fps)
    _node(fmt, "width", width)
    _node(fmt, "height", height)
    _node(fmt, "pixelaspectratio", "square")
    track = _node(video, "track")
    seen = set()
    dissolve = int(fps * 0.5)
    for i, entry in enumerate(timeline):
        clip = entry["clip"]
        start, end = entry["frame_start"], entry["frame_end"]
        use_prev = transition and i > 0
        use_next = transition and i < len(timeline) - 1
        if use_prev:
            mid = start
            head = ET.SubElement(track, "transitionitem")
            _node(head, "start", mid - dissolve // 2)
            _node(head, "end", mid + dissolve // 2)
            _node(head, "alignment", "center")
            _rate(head, fps)
            effect = _node(head, "effect")
            _node(effect, "name", "Cross Dissolve")
            _node(effect, "effectid", "Cross Dissolve")
            _node(effect, "effecttype", "transition")
            _node(effect, "mediatype", "video")
        item = _node(track, "clipitem", id=f"clipitem-{i + 1}")
        _node(item, "name", clip["name"])
        _node(item, "enabled", "TRUE")
        _node(item, "duration", int(clip["duration"] * fps))
        _rate(item, fps)
        _node(item, "start", -1 if use_prev else start)
        _node(item, "end", -1 if use_next else end)
        _node(item, "in", entry["frame_in"])
        _node(item, "out", entry["frame_in"] + end - start)
        _file_node(item, f"file-{clip['index']}", clip["path"], clip, fps, seen)
        fit = max(width / clip["width"], height / clip["height"]) * 100 * punch
        motion = _node(_node(item, "filter"), "effect")
        _node(motion, "name", "Basic Motion")
        _node(motion, "effectid", "basic")
        _node(motion, "effectcategory", "motion")
        _node(motion, "effecttype", "motion")
        _node(motion, "mediatype", "video")
        param = _node(motion, "parameter")
        _node(param, "parameterid", "scale")
        _node(param, "name", "Scale")
        _node(param, "valuemin", 0)
        _node(param, "valuemax", 10000)
        _node(param, "value", round(fit, 2))
    audio = _node(media, "audio")
    audio_track = _node(audio, "track")
    item = _node(audio_track, "clipitem", id="clipitem-song")
    _node(item, "name", Path(song).name)
    _node(item, "enabled", "TRUE")
    _node(item, "duration", int(song_info["duration"] * fps))
    _rate(item, fps)
    _node(item, "start", 0)
    _node(item, "end", total)
    _node(item, "in", 0)
    _node(item, "out", total)
    _file_node(item, "file-song", song, song_info, fps, seen)
    source = _node(item, "sourcetrack")
    _node(source, "mediatype", "audio")
    _node(source, "trackindex", 1)
    text = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent=" ")
    text = text.replace('<?xml version="1.0" ?>', '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>')
    Path(path).write_text(text, encoding="utf-8")


def render_cut(args):
    entry, index, width, height, fps, punch, transition, folder = args
    clip = entry["clip"]
    span = entry["span"]
    target = Path(folder) / f"cut_{index:05d}.mp4"
    tw, th = int(width * punch) // 2 * 2, int(height * punch) // 2 * 2
    vf = [f"scale={tw}:{th}:force_original_aspect_ratio=increase", f"crop={width}:{height}", "setsar=1", f"fps={fps}",
          "lut3d=grade.cube", "format=yuv420p"]
    if transition == "flash" and index > 0:
        vf.append(f"fade=in:st=0:d={min(0.12, span / 3):.3f}:color=white")
    elif transition == "dissolve" and index > 0:
        vf.append(f"fade=in:st=0:d={min(0.25, span / 3):.3f}")
        vf.append(f"fade=out:st={max(span - 0.25, 0):.3f}:d=0.25")
    run([tool("ffmpeg"), "-y", "-v", "error", "-ss", f"{entry['source_in']:.3f}", "-i", str(Path(clip["path"]).resolve()),
         "-t", f"{span:.3f}", "-vf", ",".join(vf), "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
         target.name], cwd=str(folder))
    return target


def edit(job, song, clips, style_key="performance", fps=30, seed=1, length=0, quality=20):
    style = STYLES[style_key]
    if not clips:
        raise RuntimeError("Video clips nahi diye.")
    title = Path(song).stem
    out = output_folder("VideoEdits", title)
    temp = out / "_temp"
    temp.mkdir()
    job.update(0.02, "Gaane ki beats nikal raha hai")
    y = decode_mono(song, 22050)
    env, rate = onset_envelope(y, 22050)
    bpm, _ = estimate_bpm(env, rate)
    beats = beat_track(env, rate, bpm)
    song_info = media_info(song)
    total = float(length) if length else song_info["duration"]
    if len(beats) < 8:
        beats = np.arange(0, total, 0.5)
    beats = np.concatenate(([0.0], beats[beats > 0.1]))
    win = 22050
    loud = np.sqrt(np.convolve(y ** 2, np.ones(win) / win, mode="same"))[::2205]
    energy = np.interp(beats * 10, np.arange(len(loud)), loud)
    energy = (energy - np.percentile(energy, 10)) / (np.percentile(energy, 95) - np.percentile(energy, 10) + 1e-9)
    song_bands = band_features(y, 22050, hop=220)

    job.update(0.06, "Clips dekh raha hai")
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [(c, song_bands) for c in clips]
        infos = []
        for i, info in enumerate(pool.map(analyse_clip, tasks), start=1):
            job.update(0.06 + 0.26 * i / len(clips), f"Clip analyse: {i}/{len(clips)}")
            info["index"] = i
            infos.append(info)
    infos = [c for c in infos if c["has_video"] and c["duration"] > 0.5]
    if not infos:
        raise RuntimeError("Koi kaam ka video clip nahi mila.")

    vertical = style_key == "reel"
    base = max(1080, min(max(c["height"] for c in infos), 2160)) if not vertical else 1080
    width, height = (base * 16 // 9 // 2 * 2, base) if not vertical else (1080, 1920)
    rng = random.Random(seed)
    timeline = []
    last = None
    # Frame numbers come first and seconds follow, so every cut stays on the frame grid.
    for start, end in build_cuts(beats, energy, style, total):
        frame_start, frame_end = int(round(start * fps)), int(round(end * fps))
        if frame_end <= frame_start:
            continue
        clip, source_in = pick(infos, frame_start / fps, frame_end / fps, style, last, rng)
        span = min((frame_end - frame_start) / fps, clip["duration"] - source_in)
        clip["used"].append((source_in, source_in + span))
        timeline.append({"clip": clip, "source_in": source_in, "span": (frame_end - frame_start) / fps,
                         "frame_start": frame_start, "frame_end": frame_end, "frame_in": int(round(source_in * fps))})
        last = clip["name"]
    write_cube(temp / "grade.cube", style["grade"])

    job.update(0.34, "Premiere Pro files bana raha hai")
    slug = safe_name(title)
    premiere = out / "Premiere_Pro"
    premiere.mkdir()
    write_premiere(premiere / f"{slug}_Edit.xml", f"{title} - {style['label']}", timeline, song, song_info, fps,
                   width, height, style["punch"], style["transition"] == "dissolve")
    if style["transition"] == "dissolve":
        write_premiere(premiere / f"{slug}_CutsOnly.xml", f"{title} - Cuts", timeline, song, song_info, fps,
                       width, height, style["punch"], False)
    write_cube(premiere / f"{style['grade']}_Look.cube", style["grade"])
    (premiere / "Kaise_Kholein.txt").write_text(
        "1. Premiere Pro mein New Project banayein.\n2. File > Import se _Edit.xml chunein (Open se nahi).\n"
        "3. Sequence open hogi. File > Save se ye .prproj ban jayega.\n4. Rang ke liye Adjustment Layer lagayein, "
        "Lumetri Color > Creative > Look mein .cube file chunein.\n5. Agar transitions wali XML import na ho to "
        "_CutsOnly.xml use karein.\n6. Clips apni jagah se move karne par Premiere link dobara maangega.\n", encoding="utf-8")

    job.update(0.36, "Video cuts bana raha hai")
    tasks = [(entry, i, width, height, fps, style["punch"], style["transition"], str(temp)) for i, entry in enumerate(timeline)]
    cuts = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, cut in enumerate(pool.map(render_cut, tasks), start=1):
            job.update(0.36 + 0.42 * i / len(tasks), f"Cut {i}/{len(tasks)}")
            cuts.append(cut)
    (temp / "list.txt").write_text("".join(f"file '{c.name}'\n" for c in cuts), encoding="utf-8")
    extra = []
    if style["bars"]:
        bar = int(height * 0.09)
        extra = ["-vf", f"drawbox=y=0:w=iw:h={bar}:color=black:t=fill,drawbox=y=ih-{bar}:w=iw:h={bar}:color=black:t=fill"]
    target = out / f"{slug}_{style_key}_Edit.mp4"
    args = ["-f", "concat", "-safe", "0", "-i", "list.txt", "-i", str(Path(song).resolve())] + extra + [
        "-map", "0:v:0", "-map", "1:a:0", "-t", f"{total:.3f}", *encoder_args(int(quality)), "-c:a", "aac", "-b:a", "320k",
        "-movflags", "+faststart", str(target.resolve())]
    ffmpeg_progress(args, total, job, 0.8, 0.98, "Final edit render", cwd=str(temp))
    for leftover in temp.iterdir():
        leftover.unlink()
    temp.rmdir()
    used = {e["clip"]["name"] for e in timeline}
    synced = [{"name": c["name"], "offset": round(c["sync"], 2), "confidence": c["sync_confidence"]}
              for c in infos if c["sync"] is not None]
    (out / "Edit_Report.json").write_text(json.dumps({"bpm": round(bpm, 1), "style": style["label"], "cuts": len(timeline),
                                                      "lip_sync_clips": synced}, indent=1), encoding="utf-8")
    return {"folder": str(out), "file": str(target), "premiere": str(premiere), "cuts": len(timeline), "bpm": round(bpm, 1),
            "clips_used": len(used), "clips_total": len(infos), "synced": len(synced)}
