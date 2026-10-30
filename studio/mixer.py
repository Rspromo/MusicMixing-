"""Automatic stem mixing and mastering engine."""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import soxr
from scipy import ndimage, signal

from .analysis import analyze_file, decode_mono, estimate_bpm, onset_envelope
from .core import output_folder, run, safe_name, tool

AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg"}
ROLES = ["lead_vocal", "double_vocal", "harmony", "kick", "snare", "hats", "cymbals", "percussion", "drums", "bass",
         "electric_guitar", "acoustic_guitar", "piano", "keys", "pad", "strings", "brass", "flute", "synth_lead",
         "fx", "other"]

KEYWORDS = [
    ("double_vocal", r"doubl|dbl|dub"),
    ("harmony", r"harm|bgv|backing|\bbv\b|chorus.?vox|choir|adlib|ad.lib|layer"),
    ("lead_vocal", r"lead|main.?vox|vocal|vox|voice|singer|\bvoc\b|\bvo\b"),
    ("kick", r"kick|\bkik\b|\bbd\b"),
    ("snare", r"snare|\bsn\b|clap|rim"),
    ("hats", r"\bhat|hihat|hi.hat|\bhh\b|shaker"),
    ("cymbals", r"cymbal|crash|\bride\b|overhead|\boh\b"),
    ("percussion", r"tabla|dhol|dholak|dhak|nagada|tumbi|perc|conga|bongo|tamb|cajon|mridang|manjira"),
    ("drums", r"drum|\bkit\b|\bbeat\b|\bloop\b"),
    ("bass", r"bass|\bsub\b|808"),
    ("acoustic_guitar", r"acou|ac.?gtr|nylon|\bag\b|mandolin|ukulele"),
    ("electric_guitar", r"guitar|gtr|elec|\beg\b|lead.?gt"),
    ("piano", r"piano|\bpno\b|grand|rhodes"),
    ("keys", r"keys|keyboard|organ|harmonium|sitar|santoor|sarod|veena"),
    ("pad", r"\bpad\b|pads|drone|tanpura|atmos|ambien"),
    ("strings", r"string|violin|viola|cello|sarangi|esraj"),
    ("brass", r"brass|horn|trumpet|sax|trombone"),
    ("flute", r"flute|bansuri|shehnai|whistle"),
    ("synth_lead", r"synth|lead.?syn|arp|pluck"),
    ("fx", r"\bfx\b|sfx|riser|sweep|impact|whoosh|transition"),
]


@dataclass
class Preset:
    level: float               # target loudness relative to the lead vocal, in LU
    hpf: float = 0
    lpf: float = 0
    comp: tuple = None         # threshold offset below peak-ish level, ratio, attack ms, release ms
    eq: list = field(default_factory=list)  # (type, freq, gain, q)
    pan: float = 0
    reverb: float = 0
    delay: float = 0
    bus: str = "music"


P = {
    "lead_vocal": Preset(0, 90, 0, (9, 3.5, 6, 90), [("peak", 300, -1.5, 1.2), ("peak", 3200, 1.5, 0.9),
                                                     ("high", 11000, 2, .7)], 0, .14, .10, "vocal"),
    "double_vocal": Preset(-8, 130, 14000, (10, 4, 5, 80), [("peak", 300, -2.5, 1.2), ("high", 10000, -1, .7)],
                           .35, .18, .05, "vocal"),
    "harmony": Preset(-9, 160, 13000, (10, 4, 5, 90), [("peak", 350, -3, 1.0), ("peak", 2800, -1.5, 1.0)],
                      .6, .28, .06, "vocal"),
    "kick": Preset(-5, 30, 0, (5, 4, 12, 70), [("peak", 65, 2, 1.0), ("peak", 370, -4, 1.3), ("peak", 3800, 2, 1.0)],
                   0, 0, 0, "drums"),
    "snare": Preset(-6.5, 90, 0, (6, 3.5, 8, 110), [("peak", 200, 1.5, 1.2), ("peak", 800, -2, 2), ("high", 7000, 2, .7)],
                    0, .10, 0, "drums"),
    "hats": Preset(-14, 350, 0, (4, 2.5, 3, 50), [("high", 9000, 1, .7)], .22, 0, 0, "drums"),
    "cymbals": Preset(-14, 300, 0, (4, 2, 10, 120), [("peak", 450, -2, 1)], 0, .04, 0, "drums"),
    "percussion": Preset(-9, 70, 0, (6, 3, 6, 90), [("peak", 300, -2, 1), ("peak", 5000, 1.5, 1)], -.2, .08, 0, "drums"),
    "drums": Preset(-4.5, 30, 0, (6, 3, 12, 120), [("peak", 350, -2.5, 1), ("high", 9000, 1.5, .7)], 0, .03, 0, "drums"),
    "bass": Preset(-6, 30, 0, (8, 4, 18, 130), [("peak", 80, 1.5, 1), ("peak", 280, -2.5, 1.2), ("peak", 800, 1, 1)],
                   0, 0, 0, "bass"),
    "electric_guitar": Preset(-9, 90, 12000, (6, 3, 12, 120), [("peak", 350, -2.5, 1.1), ("peak", 2500, -1, 1.1)],
                              .55, .10, .03),
    "acoustic_guitar": Preset(-9.5, 110, 0, (6, 3, 10, 120), [("peak", 220, -3, 1.1), ("high", 9000, 1.5, .7)],
                              -.4, .12, 0),
    "piano": Preset(-9.5, 70, 0, (5, 2.5, 18, 160), [("peak", 300, -2, 1.0), ("peak", 3000, -1, 1.1)], -.15, .14, 0),
    "keys": Preset(-11, 110, 0, (5, 2.5, 15, 150), [("peak", 350, -2, 1)], .25, .14, 0),
    "pad": Preset(-14, 160, 11000, (4, 2, 30, 250), [("peak", 2800, -2.5, .9)], 0, .22, 0),
    "strings": Preset(-11.5, 140, 13000, (4, 2, 25, 200), [("peak", 320, -2, 1)], -.1, .24, 0),
    "brass": Preset(-11, 140, 0, (5, 3, 10, 120), [("peak", 300, -2, 1)], .2, .16, 0),
    "flute": Preset(-11, 220, 0, (5, 2.5, 10, 120), [("peak", 900, -1, 1.2)], -.25, .24, .06),
    "synth_lead": Preset(-10, 150, 0, (5, 3, 8, 110), [("peak", 2800, -1.5, 1)], .15, .14, .08),
    "fx": Preset(-13, 100, 0, None, [], 0, .12, 0),
    "other": Preset(-11.5, 80, 0, (5, 2.5, 12, 130), [("peak", 320, -1.5, 1)], 0, .10, 0),
}

GENRES = {
    "bollywood": {"label": "Bollywood / Filmy", "vocal": 1.0, "reverb": 1.15, "bass": 0.0, "sidechain": 3, "lufs": -11},
    "punjabi": {"label": "Punjabi / Bhangra", "vocal": 0.5, "reverb": 0.8, "bass": 1.5, "sidechain": 4, "lufs": -9.5},
    "hiphop": {"label": "Hip-hop / Trap", "vocal": 1.0, "reverb": 0.55, "bass": 2.0, "sidechain": 4, "lufs": -9},
    "pop": {"label": "Pop", "vocal": 0.5, "reverb": 1.0, "bass": 0.5, "sidechain": 3, "lufs": -10.5},
    "devotional": {"label": "Bhajan / Devotional", "vocal": 1.5, "reverb": 1.35, "bass": -1.5, "sidechain": 1, "lufs": -13},
    "ghazal": {"label": "Ghazal / Acoustic", "vocal": 1.5, "reverb": 1.3, "bass": -1.0, "sidechain": 0, "lufs": -14},
    "rock": {"label": "Rock", "vocal": 0.0, "reverb": 0.8, "bass": 0.5, "sidechain": 1, "lufs": -10},
}


def classify_name(name):
    text = name.lower().replace("_", " ").replace("-", " ")
    for role, pattern in KEYWORDS:
        if re.search(pattern, text):
            return role
    return None


def classify_audio(path):
    """Fallback guess from the sound itself when the file name says nothing."""
    y = decode_mono(path, 22050, start=0, duration=120)
    if len(y) < 4096:
        return "other"
    active = y[np.abs(y) > 0.003]
    if len(active) < 2048:
        return "fx"
    frames = np.lib.stride_tricks.sliding_window_view(y, 2048)[::1024]
    frames = frames[np.sqrt((frames ** 2).mean(1)) > 0.003][:3000]
    if len(frames) == 0:
        return "fx"
    mag = np.abs(np.fft.rfft(frames * np.hanning(2048), axis=1)).mean(0)
    freqs = np.fft.rfftfreq(2048, 1 / 22050)
    centroid = (mag * freqs).sum() / (mag.sum() + 1e-9)
    low = mag[freqs < 150].sum() / (mag.sum() + 1e-9)
    crest = np.abs(y).max() / (np.sqrt((active ** 2).mean()) + 1e-9)
    if low > 0.5:
        return "kick" if crest > 8 else "bass"
    if centroid > 5500:
        return "hats"
    if crest > 9:
        return "percussion"
    if 500 < centroid < 2600 and crest > 3.5:
        return "lead_vocal"
    return "keys" if centroid < 1800 else "other"


def scan(folder):
    root = Path(folder)
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in AUDIO_EXT) if root.is_dir() else []
    stems = []
    for path in files:
        role = classify_name(path.stem)
        guessed = role is None
        stems.append({"path": str(path), "name": path.stem, "role": role or classify_audio(path), "guessed": guessed})
    leads = [s for s in stems if s["role"] == "lead_vocal"]
    if len(leads) > 1:
        # The clearest name stays lead; other vocal takes become harmonies.
        keep = next((s for s in leads if re.search(r"lead|main", s["name"], re.I)), leads[0])
        for stem in leads:
            if stem is not keep and re.search(r"[_\s-](\d+|b)$|alt|take", stem["name"], re.I):
                stem["role"] = "double_vocal"
    return stems


def load_stereo(path, sr):
    audio, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if file_sr != sr:
        audio = soxr.resample(audio, file_sr, sr, quality="HQ")
    audio -= audio.mean(axis=0, keepdims=True)
    return audio.T


def ensure_readable(path, temp):
    """Decode formats soundfile may not read, such as m4a, into a temporary WAV."""
    path = Path(path)
    if path.suffix.lower() in {".wav", ".aif", ".aiff", ".flac", ".ogg"}:
        return path
    target = Path(temp) / (path.stem + ".wav")
    run([tool("ffmpeg"), "-y", "-v", "error", "-i", str(path), "-c:a", "pcm_f32le", str(target)])
    return target


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def active_loudness(audio, meter, sr):
    """Integrated loudness over the audible parts only, so sparse stems are judged fairly."""
    mono = audio.mean(axis=0)
    win = int(sr * 0.1)
    if len(mono) < win * 5:
        return rms_db(mono)
    count = len(mono) // win
    power = (mono[:count * win].reshape(count, win) ** 2).mean(axis=1)
    keep = power > max(power.max() * 1e-4, 1e-8)
    if keep.sum() < 5:
        return rms_db(mono)
    chunks = audio[:, :count * win].reshape(2, count, win)[:, keep].reshape(2, -1)
    try:
        value = meter.integrated_loudness(chunks.T)
    except ValueError:
        value = -np.inf
    return value if np.isfinite(value) else rms_db(chunks)


def spectrum_db(mono, sr, n_fft=8192):
    if len(mono) < n_fft:
        return np.fft.rfftfreq(n_fft, 1 / sr), np.full(n_fft // 2 + 1, -120.0)
    freqs, power = signal.welch(mono, sr, nperseg=n_fft, noverlap=n_fft // 2)
    return freqs, 10 * np.log10(power + 1e-14)


def find_resonances(mono, sr, max_cuts=3):
    """Find narrow ringing peaks that stick out of the smoothed spectrum."""
    freqs, db = spectrum_db(mono, sr)
    band = (freqs > 150) & (freqs < 6000)
    idx = np.where(band)[0]
    smooth = signal.medfilt(db, 101)
    excess = db - smooth
    peaks, props = signal.find_peaks(excess[idx], height=7, distance=20)
    order = np.argsort(props["peak_heights"])[::-1][:max_cuts]
    return [(float(freqs[idx[peaks[o]]]), float(min(props["peak_heights"][o] * 0.5, 5))) for o in order]


def envelope(x, sr, attack=0.005, release=0.1):
    rect = np.abs(x)
    a = np.exp(-1 / (sr * attack))
    r = np.exp(-1 / (sr * release))
    # A one-pole release follower is enough; attack is handled with a short smoother.
    b, c = [1 - r], [1, -r]
    peak = np.maximum(signal.lfilter(b, c, rect) , 0)
    smooth = signal.lfilter([1 - a], [1, -a], peak)
    return smooth


def deess(audio, sr, amount=6.0):
    sos = signal.butter(4, [5000, min(10000, sr / 2 - 500)], "bandpass", fs=sr, output="sos")
    side = signal.sosfiltfilt(sos, audio, axis=1)
    env = envelope(side.mean(axis=0), sr, 0.001, 0.05)
    voiced = env[env > env.max() * 0.02] if env.max() > 0 else env
    if len(voiced) == 0:
        return audio
    threshold = np.percentile(voiced, 80)
    over = np.maximum(env / (threshold + 1e-9), 1)
    gain_db = -np.minimum(20 * np.log10(over) * 0.6, amount)
    gain = 10 ** (gain_db / 20) - 1
    return audio + side * gain


def vocal_rider(audio, sr, max_db=4.0):
    """Ride the vocal towards a steady level, like a fader move on every phrase."""
    mono = audio.mean(axis=0)
    win = int(sr * 0.05)
    count = len(mono) // win
    if count < 10:
        return audio
    level = 10 * np.log10((mono[:count * win].reshape(count, win) ** 2).mean(axis=1) + 1e-10)
    active = level > level.max() - 40
    if active.sum() < 10:
        return audio
    target = np.median(level[active])
    gain = np.where(active, np.clip(target - level, -max_db, max_db), 0.0)
    gain = signal.filtfilt(*signal.butter(2, 0.06), gain)
    curve = np.interp(np.arange(len(mono)), (np.arange(count) + 0.5) * win, gain)
    return audio * 10 ** (curve / 20)


def peak_sos(freq, gain_db, q, sr):
    """RBJ cookbook peaking filter."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * freq / sr
    alpha = np.sin(w) / (2 * q)
    b = [1 + alpha * A, -2 * np.cos(w), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w), 1 - alpha / A]
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1, a[1] / a[0], a[2] / a[0]]])


def shelf_sos(freq, gain_db, q, sr, high=True):
    """RBJ cookbook shelving filter."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * freq / sr
    cos, root = np.cos(w), 2 * np.sqrt(A) * np.sin(w) / (2 * q)
    sign = 1 if high else -1
    b = [A * ((A + 1) + sign * (A - 1) * cos + root), -2 * sign * A * ((A - 1) + sign * (A + 1) * cos),
         A * ((A + 1) + sign * (A - 1) * cos - root)]
    a = [(A + 1) - sign * (A - 1) * cos + root, 2 * sign * ((A - 1) - sign * (A + 1) * cos),
         (A + 1) - sign * (A - 1) * cos - root]
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1, a[1] / a[0], a[2] / a[0]]])


def compress(audio, sr, threshold_db, ratio, attack_ms, release_ms, knee_db=6.0):
    """Stereo-linked feed-forward compressor with a soft knee."""
    detector = np.abs(audio).max(axis=0)
    env = envelope(detector, sr, attack_ms / 1000, release_ms / 1000)
    over = 20 * np.log10(env + 1e-9) - threshold_db
    excess = np.where(over <= -knee_db / 2, 0.0,
                      np.where(over >= knee_db / 2, over, (over + knee_db / 2) ** 2 / (2 * knee_db)))
    return audio * 10 ** (-excess * (1 - 1 / ratio) / 20)


def apply_chain(audio, preset, level_db, resonances, sr):
    sections = []
    if preset.hpf:
        sections.append(signal.butter(2, preset.hpf, "highpass", fs=sr, output="sos"))
    if preset.lpf and preset.lpf < sr / 2 - 500:
        sections.append(signal.butter(2, preset.lpf, "lowpass", fs=sr, output="sos"))
    for freq, cut in resonances:
        sections.append(peak_sos(freq, -cut, 6.0, sr))
    for kind, freq, gain, q in preset.eq:
        if kind == "peak":
            sections.append(peak_sos(freq, gain, q, sr))
        else:
            sections.append(shelf_sos(freq, gain, q, sr, high=kind == "high"))
    if sections:
        audio = signal.sosfilt(np.vstack(sections), audio, axis=1)
    if preset.comp:
        below, ratio, attack, release = preset.comp
        audio = compress(audio, sr, level_db - below + 6, ratio, attack, release)
    return audio.astype(np.float32)


def plate_reverb(send, sr, room):
    """Convolve with a synthetic stereo tail whose highs die faster than its lows."""
    decay = 0.9 + 2.0 * room
    size = int(decay * sr)
    t = np.arange(size) / sr
    rng = np.random.default_rng(11)
    noise = rng.standard_normal((2, size))
    low_sos = signal.butter(2, 2500, "lowpass", fs=sr, output="sos")
    low = signal.sosfilt(low_sos, noise, axis=1)
    impulse = low * np.exp(-6.9 * t / decay) + (noise - low) * np.exp(-6.9 * t / (decay * 0.45))
    impulse *= np.minimum(t / 0.012, 1)  # a soft onset reads as distance rather than a click
    impulse /= np.sqrt((impulse ** 2).sum(axis=1, keepdims=True))
    tone = np.vstack((signal.butter(2, 250, "highpass", fs=sr, output="sos"),
                      signal.butter(2, 8000, "lowpass", fs=sr, output="sos")))
    wet = signal.sosfilt(tone, send, axis=1)
    out = np.stack([signal.oaconvolve(wet[c], impulse[c], mode="full")[:send.shape[1]] for c in (0, 1)])
    return out.astype(np.float32)


def pan_gains(pan):
    angle = (pan + 1) * np.pi / 4
    return np.cos(angle) * np.sqrt(2), np.sin(angle) * np.sqrt(2)


def assign_pans(stems):
    """Spread same-role stems to opposite sides and respect L/R names."""
    seen = {}
    for stem in stems:
        preset = P[stem["role"]]
        name = stem["name"].lower()
        if preset.pan == 0 and stem["role"] not in ("double_vocal", "harmony"):
            stem["pan"] = 0.0
        else:
            n = seen.get(stem["role"], 0)
            seen[stem["role"]] = n + 1
            stem["pan"] = preset.pan * (1 if n % 2 == 0 else -1)
        if re.search(r"[ _\-.](l|left)$", name):
            stem["pan"] = -max(abs(stem["pan"]), 0.7)
        elif re.search(r"[ _\-.](r|right)$", name):
            stem["pan"] = max(abs(stem["pan"]), 0.7)


def band_duck(music, key_env, sr, low, high, depth_db):
    """Dip one band of the music only while the key signal is present."""
    if depth_db <= 0:
        return music
    sos = signal.butter(2, [low, high], "bandpass", fs=sr, output="sos")
    band = signal.sosfiltfilt(sos, music, axis=1)
    gain = 10 ** (-depth_db * key_env / 20) - 1
    return music + band * gain


def normalized_env(x, sr, attack, release):
    env = envelope(x, sr, attack, release)
    top = np.percentile(env, 99)
    return np.clip(env / (top + 1e-9), 0, 1)


def tempo_delay(signal_in, sr, bpm):
    beat = 60 / (bpm if bpm > 0 else 100)
    steps = int(beat * 0.75 * sr)
    out = np.zeros_like(signal_in)
    if steps <= 0 or steps >= signal_in.shape[1]:
        return out
    for i, level in enumerate((0.55, 0.3, 0.16), start=1):
        off = steps * i
        pan = (1.0, 0.55) if i % 2 else (0.55, 1.0)
        out[0, off:] += signal_in[0, :-off] * level * pan[0]
        out[1, off:] += signal_in[1, :-off] * level * pan[1]
    sos = signal.butter(2, [300, 4500], "bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, out, axis=1)


def smooth_match_eq(mix, reference, sr, limit=3.0):
    """Match the long-term tonal balance of a reference through a limited FIR."""
    def mono(a):
        return a.mean(axis=0)
    freqs, a = spectrum_db(mono(mix), sr)
    _, b = spectrum_db(mono(reference), sr)
    centers = np.geomspace(40, 16000, 28)
    width = 2 ** (1 / 3)
    diff = []
    for c in centers:
        mask = (freqs >= c / width) & (freqs < c * width)
        diff.append(np.mean(b[mask] - a[mask]) if mask.any() else 0)
    diff = np.array(diff) - np.median(diff)
    diff = np.clip(signal.medfilt(diff, 3) * 0.7, -limit, limit)
    grid = np.linspace(0, sr / 2, 2049)
    gains = np.interp(np.log10(np.maximum(grid, 20)), np.log10(centers), diff, left=0, right=0)
    fir = signal.firwin2(4095, grid / (sr / 2), 10 ** (gains / 20))
    wet = signal.fftconvolve(mix, fir[None, :], mode="same", axes=1)
    return wet, dict(zip([int(c) for c in centers], np.round(diff, 1).tolist()))


def true_peak_db(audio, sr):
    resampled = soxr.resample(audio.T, sr, sr * 4, quality="HQ") if len(audio.T) else audio.T
    return 20 * np.log10(np.abs(resampled).max() + 1e-9)


def glue_gain(bus, sr, ratio=2.0):
    """Gain curve of a slow 2:1 bus compressor sitting about 2 dB over the bus level."""
    mono = np.abs(bus).max(axis=0)
    env = envelope(mono, sr, 0.025, 0.12)
    audible = env[env > env.max() * 0.01]
    if len(audible) == 0:
        return np.ones(len(mono), dtype=np.float32)
    threshold = np.sqrt(np.mean(audible ** 2)) * 10 ** (2 / 20)
    over = 20 * np.log10(np.maximum(env / threshold, 1.0))
    return (10 ** (-over * (1 - 1 / ratio) / 20)).astype(np.float32)


def brickwall(audio, sr, ceiling_db, lookahead_ms=3.0, release_ms=90.0):
    """Lookahead limiter: gain is planned from the whole file, so no peak can pass the ceiling."""
    ceiling = 10 ** (ceiling_db / 20)
    peak = np.abs(audio).max(axis=0)
    need = 1 - np.minimum(ceiling / np.maximum(peak, 1e-9), 1.0)
    window = max(int(sr * lookahead_ms / 1000), 1) | 1
    hold = ndimage.maximum_filter1d(need, window * 2 + 1)
    pole = np.exp(-1 / (sr * release_ms / 1000))
    released = signal.lfilter([1 - pole], [1, -pole], hold)
    reduction = np.maximum(hold, released)
    # The box smoother stays inside the held region, so full reduction is kept on each peak.
    smooth = ndimage.uniform_filter1d(reduction, window)
    return np.clip(audio * (1 - smooth), -ceiling, ceiling)


def master(mix, sr, lufs, ceiling=-1.0, reference=None):
    notes = {}
    mix = mix.copy()
    if reference is not None:
        mix, notes["reference_eq_db"] = smooth_match_eq(mix, reference, sr)
    # Mono the deep lows so bass stays solid on speakers and phones.
    sos = signal.butter(4, 120, "lowpass", fs=sr, output="sos")
    low = signal.sosfiltfilt(sos, mix, axis=1)
    mono_low = low.mean(axis=0, keepdims=True)
    mix = mix - low + mono_low
    peak = np.abs(mix).max()
    if peak > 0:
        mix = mix / peak * 10 ** (-6 / 20)
    air = np.vstack((signal.butter(2, 25, "highpass", fs=sr, output="sos"), shelf_sos(12000, 0.6, 0.7, sr)))
    mix = compress(signal.sosfilt(air, mix, axis=1), sr, -16, 1.8, 30, 200).astype(np.float32)
    meter = pyln.Meter(sr)
    limit_at = ceiling - 0.2  # a margin for inter-sample overshoot
    gain = 0.0
    result = mix
    for _ in range(8):
        result = brickwall((mix * 10 ** (gain / 20)).astype(np.float32), sr, limit_at)
        now = meter.integrated_loudness(result.T)
        if not np.isfinite(now) or abs(now - lufs) < 0.1:
            break
        gain = min(gain + float(np.clip(lufs - now, -6, 6)), 24)
    tp = true_peak_db(result, sr)
    for _ in range(5):
        if tp <= ceiling:
            break
        limit_at -= tp - ceiling + 0.05
        result = brickwall((mix * 10 ** (gain / 20)).astype(np.float32), sr, limit_at)
        tp = true_peak_db(result, sr)
    if tp > ceiling:
        result = result * 10 ** ((ceiling - tp) / 20)
        tp = ceiling
    notes.update(lufs=round(float(meter.integrated_loudness(result.T)), 2), true_peak_db=round(float(tp), 2),
                 limiter_push_db=round(float(gain), 1))
    return result, notes


def write_audio(path, audio, sr):
    sf.write(str(path), np.clip(audio, -1, 1).T, sr, subtype="PCM_24")


def write_mp3(wav, target):
    run([tool("ffmpeg"), "-y", "-v", "error", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "320k", str(target)])


def mix_song(job, stems, name="Song", genre="bollywood", reference=None, lufs=None, vocal_boost=0.0,
             reverb_amount=1.0, variants=True, export_stems=True):
    if not stems:
        raise RuntimeError("Koi stem nahi mila.")
    settings = GENRES.get(genre, GENRES["bollywood"])
    out = output_folder("Mixes", name)
    temp = out / "_temp"
    temp.mkdir()
    job.update(0.02, "Stems load ho rahe hain")
    readable = [ensure_readable(s["path"], temp) for s in stems]
    srs = [sf.info(str(p)).samplerate for p in readable]
    sr = max(max(srs), 44100)
    if sr > 48000:
        sr = 48000  # keeps a 7 GB machine safe; the master is unaffected in practice
    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded = list(pool.map(lambda p: load_stereo(p, sr), readable))
    length = max(a.shape[1] for a in loaded)
    tail = int(sr * 2)
    stems = [dict(s) for s in stems]
    assign_pans(stems)
    for stem, audio in zip(stems, loaded):
        pad = length + tail - audio.shape[1]
        stem["audio"] = np.pad(audio, ((0, 0), (0, pad)))

    job.update(0.1, "Tempo aur Key dhoondh raha hai")
    drums = [s for s in stems if P[s["role"]].bus == "drums"] or stems
    probe_mono = sum(s["audio"].mean(axis=0) for s in drums)
    y22 = soxr.resample(probe_mono, sr, 22050)
    bpm = estimate_bpm(*onset_envelope(y22, 22050))[0]

    report = {"name": name, "genre": settings["label"], "sample_rate": sr, "bpm": round(bpm, 1), "tracks": []}
    meter = pyln.Meter(sr)
    leads = [s for s in stems if s["role"] == "lead_vocal"]
    reference_lufs = -20.0  # working level of the lead vocal, which leaves healthy headroom
    processed = []

    def process(stem):
        role = stem["role"]
        preset = P[role]
        audio = stem["audio"]
        peak = np.abs(audio).max()
        if peak < 1e-5:
            return None
        audio = audio / peak * 0.5
        mono = audio.mean(axis=0)
        resonances = find_resonances(mono, sr) if role not in ("kick", "hats", "cymbals", "fx") else []
        level = rms_db(mono[np.abs(mono) > 0.001]) if (np.abs(mono) > 0.001).any() else -40
        if role == "lead_vocal":
            audio = vocal_rider(audio, sr)
        wet = apply_chain(audio, preset, level, resonances, sr)
        if role in ("lead_vocal", "double_vocal", "harmony"):
            wet = deess(wet, sr)
        loud = active_loudness(wet, meter, sr)
        target = reference_lufs + preset.level
        if role == "bass":
            target += settings["bass"]
        gain = np.clip(target - loud, -30, 24)
        wet = wet * 10 ** (gain / 20)
        left, right = pan_gains(stem["pan"])
        wet = np.stack((wet[0] * left, wet[1] * right)).astype(np.float32)
        stem["audio"] = None
        info = {"name": stem["name"], "role": role, "gain_db": round(float(gain), 1), "pan": round(float(stem["pan"]), 2),
                "hpf_hz": preset.hpf, "lpf_hz": preset.lpf, "resonance_cuts": [(round(f), round(c, 1)) for f, c in resonances],
                "eq": preset.eq, "compressor": {"threshold_db": round(level - preset.comp[0] + 6, 1), "ratio": preset.comp[1],
                                                "attack_ms": preset.comp[2], "release_ms": preset.comp[3]} if preset.comp else None,
                "reverb_send": preset.reverb, "delay_send": preset.delay}
        return {"stem": stem, "audio": wet, "info": info}

    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(process, stems):
            done += 1
            job.update(0.12 + 0.43 * done / len(stems), f"Track process: {done}/{len(stems)}")
            if result:
                processed.append(result)
    if not processed:
        raise RuntimeError("Saare stems khaali (silent) hain.")
    if not leads:
        report["warning"] = "Lead vocal stem nahi mila, isliye balance sirf instruments par hai."

    job.update(0.58, "Kick-bass aur vocal ki jagah bana raha hai")
    n = processed[0]["audio"].shape[1]
    kicks = [p for p in processed if p["info"]["role"] in ("kick", "drums")]
    vocals = [p for p in processed if p["stem"]["role"] == "lead_vocal"]
    if kicks and settings["sidechain"]:
        key = normalized_env(sum(k["audio"].mean(axis=0) for k in kicks), sr, 0.002, 0.09)
        for item in processed:
            if item["info"]["role"] == "bass":
                item["audio"] = band_duck(item["audio"], key, sr, 40, 180, settings["sidechain"]).astype(np.float32)
    if vocals:
        key = normalized_env(sum(v["audio"].mean(axis=0) for v in vocals), sr, 0.02, 0.25)
        for item in processed:
            if P[item["info"]["role"]].bus == "music":
                item["audio"] = band_duck(item["audio"], key, sr, 1500, 4500, 2.5).astype(np.float32)

    job.update(0.65, "Reverb aur delay laga raha hai")
    beat = 60 / (bpm if bpm > 0 else 100)
    send_rev = np.zeros((2, n), dtype=np.float32)
    send_dly = np.zeros((2, n), dtype=np.float32)
    amount = settings["reverb"] * reverb_amount
    for item in processed:
        preset = P[item["info"]["role"]]
        send_rev += item["audio"] * preset.reverb * amount
        send_dly += item["audio"] * preset.delay * amount
    predelay = int(min(beat / 8, 0.04) * sr)
    room = np.clip(0.45 + beat * 0.15, 0.4, 0.75)
    rev_in = np.pad(send_rev, ((0, 0), (predelay, 0)))[:, :n]
    fx_reverb = plate_reverb(rev_in, sr, float(room))
    fx_delay = tempo_delay(send_dly, sr, bpm).astype(np.float32)
    if vocals:
        # Keep vocal ambience out of the way while the vocal is sounding.
        key = normalized_env(sum(v["audio"].mean(axis=0) for v in vocals), sr, 0.01, 0.2)
        fx_delay = fx_delay * (10 ** (-6 * key / 20))

    job.update(0.72, "Buses aur balance")
    buses = {"vocal": np.zeros((2, n), dtype=np.float32), "drums": np.zeros((2, n), dtype=np.float32),
             "bass": np.zeros((2, n), dtype=np.float32), "music": np.zeros((2, n), dtype=np.float32)}
    drum_items = [p for p in processed if P[p["info"]["role"]].bus == "drums"]
    if drum_items:
        # Glue is one gain curve applied per stem, so exported stems still sum to the mix.
        curve = glue_gain(sum(p["audio"] for p in drum_items), sr)
        for item in drum_items:
            item["audio"] = (item["audio"] * curve).astype(np.float32)
    for item in processed:
        buses[P[item["info"]["role"]].bus] += item["audio"]
        report["tracks"].append(item["info"])
    voc_gain = 10 ** ((settings["vocal"] + vocal_boost) / 20)
    ambience = fx_reverb + fx_delay
    instrumental = buses["drums"] + buses["bass"] + buses["music"]
    full = instrumental + buses["vocal"] * voc_gain + ambience

    ref_audio = None
    if reference:
        job.update(0.75, "Reference gaana match kar raha hai")
        ref_path = ensure_readable(reference, temp)
        ref_audio = load_stereo(ref_path, sr)
    target = lufs if lufs else settings["lufs"]

    job.update(0.78, "Mastering")
    slug = safe_name(name)
    stereo, notes = master(full, sr, target, reference=ref_audio)
    report["master"] = notes
    mix_path = out / f"{slug}_MASTER.wav"
    write_audio(mix_path, stereo, sr)
    premaster = out / f"{slug}_MIX_premaster.wav"
    peak = np.abs(full).max()
    write_audio(premaster, full / max(peak, 1e-9) * 0.5, sr)
    write_mp3(mix_path, out / f"{slug}_MASTER.mp3")

    if variants:
        job.update(0.84, "Vocal Up aur Instrumental versions")
        version, _ = master(instrumental + buses["vocal"] * voc_gain * 10 ** (1.5 / 20) + ambience, sr, target, reference=ref_audio)
        write_audio(out / f"{slug}_MASTER_VocalUp.wav", version, sr)
        version, _ = master(instrumental + fx_reverb * 0.5, sr, target, reference=ref_audio)
        write_audio(out / f"{slug}_MASTER_Instrumental.wav", version, sr)

    if export_stems:
        job.update(0.9, "Ableton ke liye processed stems likh raha hai")
        folder = out / "Stems_for_Ableton"
        folder.mkdir()
        # One shared gain so the stems added up in Ableton equal the premaster file.
        scale = 0.5 / max(peak, 1e-9)
        for i, item in enumerate(processed, start=1):
            gain = voc_gain if P[item["info"]["role"]].bus == "vocal" else 1.0
            write_audio(folder / f"{i:02d}_{safe_name(item['stem']['name'])}.wav", item["audio"] * gain * scale, sr)
        write_audio(folder / f"{len(processed) + 1:02d}_FX_Reverb.wav", fx_reverb * scale, sr)
        write_audio(folder / f"{len(processed) + 2:02d}_FX_Delay.wav", fx_delay * scale, sr)

    for leftover in temp.iterdir():
        leftover.unlink()
    temp.rmdir()
    (out / "Mix_Report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float),
                                         encoding="utf-8")
    return {"folder": str(out), "master": str(mix_path), "lufs": notes["lufs"], "true_peak": notes["true_peak_db"],
            "bpm": report["bpm"], "tracks": len(processed), "warning": report.get("warning", "")}


def analyze_reference(path):
    return analyze_file(path)
