"""Melodyne-style automatic note correction that keeps vibrato and breath."""
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf
from scipy import ndimage, signal

from .analysis import NOTES, analyze_file
from .core import output_folder, safe_name

FRAME_MS = 5.0
SCALES = {
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Minor": [0, 2, 3, 5, 7, 8, 10],
    "Harmonic Minor": [0, 2, 3, 5, 7, 8, 11],
    "Chromatic": list(range(12)),
    "Bilawal (Major)": [0, 2, 4, 5, 7, 9, 11],
    "Yaman (Kalyan)": [0, 2, 4, 6, 7, 9, 11],
    "Khamaj": [0, 2, 4, 5, 7, 9, 10],
    "Kafi": [0, 2, 3, 5, 7, 9, 10],
    "Asavari": [0, 2, 3, 5, 7, 8, 10],
    "Bhairav": [0, 1, 4, 5, 7, 8, 11],
    "Bhairavi": [0, 1, 3, 5, 7, 8, 10],
    "Todi": [0, 1, 3, 6, 7, 8, 11],
    "Purvi": [0, 1, 4, 6, 7, 8, 11],
    "Marwa": [0, 1, 4, 6, 7, 9, 11],
}
PRESETS = {
    "natural": {"strength": 0.8, "drift": 0.4, "retune_ms": 70, "ornament": 0.3},
    "tight": {"strength": 0.95, "drift": 0.7, "retune_ms": 45, "ornament": 0.45},
    "hard": {"strength": 1.0, "drift": 1.0, "retune_ms": 15, "ornament": 1.0},
}


def find_phrases(mono, sr, gap=0.2):
    """Split on silence so each sung phrase can be tuned on its own core."""
    win = int(sr * 0.01)
    count = len(mono) // win
    if count == 0:
        return []
    level = np.sqrt((mono[:count * win].reshape(count, win) ** 2).mean(axis=1))
    top = np.percentile(level, 99)
    active = level > max(top * 0.01, 1e-4)
    active = ndimage.binary_closing(active, np.ones(int(gap * 100)))
    active = ndimage.binary_dilation(active, np.ones(11))
    labels, total = ndimage.label(active)
    phrases = []
    for slice_ in ndimage.find_objects(labels):
        start, stop = slice_[0].start * win, min(slice_[0].stop * win, len(mono))
        # Very long phrases are cut so that memory use stays predictable.
        while stop - start > 30 * sr:
            phrases.append((start, start + 25 * sr))
            start += 25 * sr
        if stop - start > sr * 0.05:
            phrases.append((start, stop))
    return phrases


def segment_notes(midi):
    """Group voiced frames into notes; a sustained jump starts a new one."""
    notes = []
    voiced = np.where(midi > 0)[0]
    if len(voiced) == 0:
        return notes
    runs = np.split(voiced, np.where(np.diff(voiced) > 3)[0] + 1)
    for run in runs:
        start = run[0]
        i = start
        pitch = ndimage.median_filter(midi[run[0]:run[-1] + 1], size=9, mode="nearest")
        offset = run[0]
        center = pitch[0]
        streak = 0
        for k in range(len(pitch)):
            if abs(pitch[k] - center) > 0.75:
                streak += 1
                if streak >= 7:
                    end = offset + k - streak
                    if end > start:
                        notes.append((start, end))
                    start = end + 1
                    center = pitch[k]
                    streak = 0
            else:
                streak = 0
                span = max(0, offset + k - start)
                center = np.median(pitch[max(start - offset, 0):k + 1]) if span < 60 else center
            i = offset + k
        if i > start:
            notes.append((start, i))
    return notes


def nearest_scale(pitch, allowed):
    octave = np.floor(pitch / 12)
    options = np.array([octave * 12 + a + n for n in (-12, 0, 12) for a in allowed])
    return float(options[np.argmin(np.abs(options - pitch))])


def tune_phrase(args):
    index, audio, sr, allowed, cfg, max_shift = args
    x = audio.astype(np.float64)
    period = FRAME_MS
    f0, times = pyworld.harvest(x, sr, f0_floor=70, f0_ceil=1100, frame_period=period)
    f0 = pyworld.stonemask(x, f0, times, sr)
    midi = np.zeros_like(f0)
    voiced = f0 > 0
    midi[voiced] = 69 + 12 * np.log2(f0[voiced] / 440)
    shift = np.zeros_like(f0)
    records = []
    for start, end in segment_notes(midi):
        length = end - start + 1
        if length * period < 70:
            continue  # scoops and passing tones stay as sung
        chunk = midi[start:end + 1]
        mask = chunk > 0
        if mask.sum() < 8:
            continue
        pitch = chunk[mask]
        center = float(np.median(pitch))
        target = nearest_scale(center, allowed)
        delta = target - center
        if abs(delta) > max_shift:
            continue
        spread = float(np.percentile(pitch, 95) - np.percentile(pitch, 5))
        strength = cfg["strength"] * (cfg["ornament"] if spread > 1.3 else 1.0)  # meend and gamak survive
        slow = chunk.copy()
        if length > 40 and mask.all():
            b, a = signal.butter(2, 2.0 / (500 / period))
            slow = signal.filtfilt(b, a, chunk)
            drift = (center - slow) * cfg["drift"] * (cfg["ornament"] if spread > 1.3 else 1.0)
        else:
            drift = np.zeros_like(chunk)
        shift[start:end + 1] = delta * strength + drift
        after = chunk + shift[start:end + 1]
        records.append({"t": float(times[start]), "len": float(length * period / 1000), "note": int(round(target)),
                        "before": round((center - target) * 100), "after": round((float(np.median(after[mask])) - target) * 100)})
    if not records:
        return index, None, times, midi, midi, records
    # A short smoother is the retune speed: slower keeps the glides into notes human.
    size = max(int(cfg["retune_ms"] / period), 1)
    smooth = ndimage.uniform_filter1d(shift, size) if size > 1 else shift
    new_f0 = f0 * 2 ** (smooth / 12)
    new_f0[~voiced] = 0
    sp = pyworld.cheaptrick(x, f0, times, sr)
    ap = pyworld.d4c(x, f0, times, sr)
    out = pyworld.synthesize(new_f0, sp, ap, sr, period)[:len(x)]
    if len(out) < len(x):
        out = np.pad(out, (0, len(x) - len(out)))
    # Only corrected, voiced sound is replaced, so consonants and breaths stay original.
    need = (np.abs(smooth) > 0.01) & voiced
    need = ndimage.binary_dilation(need, np.ones(5))
    mask = ndimage.uniform_filter1d(need.astype(np.float64), 5)
    blend = np.interp(np.arange(len(x)) / sr, times, mask)
    result = x * (1 - blend) + out * blend
    after = midi.copy()
    after[voiced] += smooth[voiced]
    return index, result.astype(np.float32), times, midi, after, records


def tune_file(job, path, allowed, cfg, out_dir, base, span, max_shift, sample_curve):
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    side = (audio[:, 0] - audio[:, 1]) / 2 if audio.shape[1] > 1 else None
    phrases = find_phrases(mono, sr)
    tuned = mono.copy()
    tasks = [(i, mono[s:e].copy(), sr, allowed, cfg, max_shift) for i, (s, e) in enumerate(phrases)]
    records = []
    curve = []
    workers = max(1, min((os.cpu_count() or 2) - 1, 8))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(tune_phrase, t) for t in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            job.update(base + span * done / max(len(tasks), 1), f"{Path(path).name}: phrase {done}/{len(tasks)}")
            index, result, times, before, after, recs = future.result()
            start = phrases[index][0]
            if result is not None:
                tuned[start:start + len(result)] = result
            offset = start / sr
            for rec in recs:
                rec["t"] = round(rec["t"] + offset, 3)
            records.extend(recs)
            if sample_curve:
                step = 4
                curve.extend([round(float(t + offset), 3), round(float(b), 2), round(float(a), 2)]
                             for t, b, a in zip(times[::step], before[::step], after[::step]) if b > 0)
    records.sort(key=lambda r: r["t"])
    curve.sort(key=lambda r: r[0])
    final = tuned if side is None else np.stack((tuned + side, tuned - side), axis=1)
    target = out_dir / f"{safe_name(Path(path).stem)}_TUNED.wav"
    sf.write(str(target), np.clip(final, -1, 1), sr, subtype="PCM_24")
    before = np.array([abs(r["before"]) for r in records]) if records else np.zeros(1)
    after = np.array([abs(r["after"]) for r in records]) if records else np.zeros(1)
    summary = {"file": str(target), "notes": len(records), "avg_error_before": round(float(before.mean()), 1),
               "avg_error_after": round(float(after.mean()), 1)}
    return summary, records, curve


def tune(job, vocals, music=None, key=None, scale=None, preset="natural", strength=None, max_shift=1.3):
    if not vocals:
        raise RuntimeError("Koi vocal file nahi di.")
    job.update(0.02, "Key aur Scale dhoondh raha hai")
    source = music or vocals[0]
    facts = analyze_file(source)
    root = key or facts["root"] or "C"
    if scale in (None, "", "auto"):
        scale = facts["mode"] or "Major"
    bpm = facts["bpm"]
    allowed = [(NOTES.index(root) + step) % 12 for step in SCALES[scale]]
    cfg = dict(PRESETS.get(preset, PRESETS["natural"]))
    if strength is not None:
        cfg["strength"] = float(strength)
    out = output_folder("Tuned", Path(vocals[0]).stem)
    results, chart = [], {}
    span = 0.92 / len(vocals)
    for i, path in enumerate(vocals):
        summary, records, curve = tune_file(job, path, allowed, cfg, out, 0.05 + i * span, span, max_shift, i == 0)
        results.append(summary)
        (out / f"{Path(path).stem}_notes.json").write_text(json.dumps(records), encoding="utf-8")
        if i == 0:
            chart = {"curve": curve[:6000], "allowed": allowed}
    return {"folder": str(out), "key": f"{root} {scale}", "bpm": bpm, "files": results, "chart": chart}
