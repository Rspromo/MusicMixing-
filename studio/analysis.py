"""Audio loading plus BPM, beat and key detection built on numpy and scipy."""
import subprocess

import numpy as np
from scipy import signal

from .core import CREATE_NO_WINDOW, tool

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def decode_mono(path, sr=22050, start=None, duration=None):
    """Decode any audio or video file to mono float32 through ffmpeg."""
    cmd = [tool("ffmpeg"), "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW)
    if proc.returncode != 0:
        raise RuntimeError(f"Audio nahi padh paya: {path}\n{proc.stderr.decode('utf-8', 'replace')[-400:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def band_features(y, sr, hop=512, n_fft=2048):
    """Log energy in 24 bands per frame."""
    window = np.hanning(n_fft)
    count = max((len(y) - n_fft) // hop, 0)
    if count == 0:
        return np.zeros((1, 24), dtype=np.float32)
    bins = n_fft // 2 + 1
    mel_edges = np.geomspace(30, min(11000, sr / 2 - 1), 25)
    edges = np.minimum((mel_edges / (sr / n_fft)).astype(int), bins - 1)
    bands = np.zeros((count, 24), dtype=np.float32)
    for start in range(0, count, 2000):
        idx = np.arange(start, min(start + 2000, count))
        frames = np.stack([y[i * hop:i * hop + n_fft] for i in idx]) * window
        mag = np.abs(np.fft.rfft(frames, axis=1))
        cum = np.cumsum(mag, axis=1)
        for b in range(24):
            lo, hi = edges[b], max(edges[b + 1], edges[b] + 1)
            bands[idx, b] = cum[:, hi] - cum[:, lo]
    return np.log1p(bands * 10)


def onset_from_bands(logged):
    flux =np.maximum(np.diff(logged, axis=0, prepend=logged[:1]), 0).sum(axis=1)
    flux -= signal.medfilt(flux, 31)
    flux = np.maximum(flux, 0)
    peak = flux.max()
    return flux / peak if peak > 0 else flux


def onset_envelope(y, sr, hop=512, n_fft=2048):
    return onset_from_bands(band_features(y, sr, hop, n_fft)), sr / hop


def estimate_bpm(env, rate, low=60, high=200):
    n = len(env)
    if n < rate * 4:
        return 0.0, 0.0
    spec = np.fft.rfft(env - env.mean(), n * 2)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
    if ac[0] <= 0:
        return 0.0, 0.0
    ac /= ac[0]
    lags = np.arange(int(rate * 60 / high), min(int(rate * 60 / low) + 1, n))
    # Bias towards a typical pop tempo, then let harmonic lags reinforce the true beat.
    bpm_axis = 60 * rate / lags
    prior = np.exp(-0.5 * (np.log2(bpm_axis / 110) / 0.9) ** 2)
    score = np.array([ac[lag] + 0.5 * ac[min(lag * 2, n - 1)] + 0.25 * ac[min(lag * 4, n - 1)] for lag in lags])
    best = int(np.argmax(score * prior))
    lag = float(lags[best])
    if 0 < best < len(lags) - 1:
        a, b, c = score[best - 1], score[best], score[best + 1]
        denom = a - 2 * b + c
        if denom != 0:
            lag += 0.5 * (a - c) / denom
    confidence = float(np.clip(ac[int(lags[best])] * 2, 0, 1))
    return float(60 * rate / lag), confidence


def beat_track(env, rate, bpm):
    """Dynamic-programming beat tracker in the style of Ellis 2007."""
    if bpm <= 0 or len(env) < 4:
        return np.array([])
    period = rate * 60 / bpm
    score = env.astype(np.float64).copy()
    back = -np.ones(len(env), dtype=int)
    lo, hi = int(round(period * 0.5)), int(round(period * 2))
    offsets = np.arange(lo, hi + 1)
    penalty = -100 * 0.01 * np.log(offsets / period) ** 2 * 100
    for i in range(hi, len(env)):
        cand = score[i - offsets] + penalty
        k = int(np.argmax(cand))
        if cand[k] > 0:
            score[i] += cand[k]
            back[i] = i - offsets[k]
    tail = int(np.argmax(score[-int(hi):])) + len(env) - int(hi)
    beats = []
    while tail >= 0:
        beats.append(tail)
        tail = back[tail]
    return np.array(beats[::-1], dtype=float) / rate


def detect_key(y, sr):
    """Return key name, mode and confidence from a whole-song chroma."""
    n_fft = 8192
    hop = 4096
    if len(y) < n_fft:
        return "", "", 0.0
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    valid = (freqs > 55) & (freqs < 2000)
    pitch_class = (np.round(12 * np.log2(freqs[valid] / 440)).astype(int) + 9) % 12
    window = np.hanning(n_fft)
    chroma = np.zeros(12)
    count = (len(y) - n_fft) // hop
    step = max(count // 1500, 1)
    for i in range(0, count, step):
        frame = y[i * hop:i * hop + n_fft] * window
        mag = np.abs(np.fft.rfft(frame))[valid] ** 2
        loudness = mag.sum()
        if loudness > 1e-8:
            frame_chroma = np.bincount(pitch_class, weights=mag, minlength=12)
            chroma += frame_chroma / loudness * np.sqrt(loudness)
    if chroma.sum() <= 0:
        return "", "", 0.0
    best, second = (-2, "", ""), -2
    for root in range(12):
        for name, profile in (("Major", MAJOR), ("Minor", MINOR)):
            corr = np.corrcoef(chroma, np.roll(profile, root))[0, 1]
            if corr > best[0]:
                second = best[0]
                best = (corr, NOTES[root], name)
            elif corr > second:
                second = corr
    return best[1], best[2], float(np.clip((best[0] - second) * 5, 0, 1))


def analyze_file(path, want_beats=False):
    y = decode_mono(path, 22050)
    if len(y) == 0:
        return {"bpm": 0, "key": "", "duration": 0}
    env, rate = onset_envelope(y, 22050)
    bpm, conf = estimate_bpm(env, rate)
    key, mode, key_conf = detect_key(y, 22050)
    result = {"bpm": round(bpm, 1), "bpm_confidence": round(conf, 2), "key": f"{key} {mode}".strip(),
              "root": key, "mode": mode, "key_confidence": round(key_conf, 2), "duration": len(y) / 22050}
    if want_beats:
        result["beats"] = beat_track(env, rate, bpm).tolist()
    return result
