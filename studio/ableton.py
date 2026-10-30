"""Read an Ableton Live Set (.als) and bounce its audio tracks to aligned stems."""
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .core import output_folder, safe_name


def _value(node, path, default=None):
    found = node.find(path)
    return found.get("Value", default) if found is not None else default


def _sample_path(clip, root):
    ref = clip.find(".//SampleRef/FileRef")
    if ref is None:
        return None
    absolute = _value(ref, "Path")
    relative = _value(ref, "RelativePath", "")
    candidates = []
    if absolute:
        candidates.append(Path(absolute))
    if relative:
        candidates.append(root / relative)
    for path in candidates:
        if path.exists():
            return path
    # The project may have been copied from another PC, so fall back to the file name.
    name = Path(absolute or relative).name if (absolute or relative) else None
    if name:
        hit = next(root.rglob(name), None)
        if hit:
            return hit
    return None


def read_project(path):
    path = Path(path)
    with gzip.open(path, "rb") as handle:
        tree = ET.fromstring(handle.read())
    tempo = float(_value(tree, ".//MainTrack//Tempo/Manual") or _value(tree, ".//MasterTrack//Tempo/Manual")
                  or _value(tree, ".//Tempo/Manual", "120"))
    tracks, warnings = [], []
    for track in tree.iter("AudioTrack"):
        name = _value(track, "Name/EffectiveName") or _value(track, "Name/UserName") or "Track"
        clips = []
        for clip in track.iter("AudioClip"):
            if clip.find("CurrentStart") is None:
                continue
            sample = _sample_path(clip, path.parent)
            if sample is None:
                warnings.append(f"{name}: audio file nahi mili")
                continue
            warped = _value(clip, "IsWarped") == "true"
            if warped:
                warnings.append(f"{name}: warped clip, timing dobara check karein")
            clips.append({"file": sample, "start": float(_value(clip, "CurrentStart", "0")),
                          "end": float(_value(clip, "CurrentEnd", "0")),
                          "offset": float(_value(clip, "Loop/LoopStart", "0")), "warped": warped})
        if clips:
            tracks.append({"name": name, "clips": clips})
    return {"tempo": tempo, "tracks": tracks, "warnings": sorted(set(warnings))}


def bounce(job, als_path):
    """Lay each track's clips on one timeline and write one WAV per track."""
    project = read_project(als_path)
    if not project["tracks"]:
        raise RuntimeError("Is project mein Arrangement ke audio clips nahi mile.")
    sec_per_beat = 60 / project["tempo"]
    out = output_folder("Ableton_Stems", Path(als_path).stem)
    sr = 48000
    span = max(c["end"] for t in project["tracks"] for c in t["clips"]) * sec_per_beat
    total = int((span + 1) * sr)
    files = []
    for i, track in enumerate(project["tracks"]):
        job.update(0.05 + 0.9 * i / len(project["tracks"]), f"Ableton track: {track['name']}")
        buffer = np.zeros((total, 2), dtype=np.float32)
        for clip in track["clips"]:
            audio, file_sr = sf.read(str(clip["file"]), always_2d=True, dtype="float32")
            if audio.shape[1] == 1:
                audio = np.repeat(audio, 2, axis=1)
            audio = audio[:, :2]
            if file_sr != sr:
                audio = soxr.resample(audio, file_sr, sr, quality="HQ")
            offset = clip["offset"] * sec_per_beat if clip["warped"] else clip["offset"]
            part = audio[int(offset * sr):]
            at = int(clip["start"] * sec_per_beat * sr)
            size = int((clip["end"] - clip["start"]) * sec_per_beat * sr)
            part = part[:min(size, total - at)]
            buffer[at:at + len(part)] += part
        target = out / f"{i + 1:02d}_{safe_name(track['name'])}.wav"
        sf.write(str(target), np.clip(buffer, -1, 1), sr, subtype="PCM_24")
        files.append(str(target))
    return {"folder": str(out), "files": files, "bpm": project["tempo"], "warnings": project["warnings"]}
