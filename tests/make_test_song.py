"""Generate a synthetic multitrack song used to exercise the engines."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000
BPM = 100
BEAT = 60 / BPM
BARS = 16
LENGTH = int(BARS * 4 * BEAT * SR)
ROOT = 220.0  # A minor
rng = np.random.default_rng(7)


def note(freq, dur, harmonics=6, decay=2.0):
    t = np.arange(int(dur * SR)) / SR
    wave = sum(np.sin(2 * np.pi * freq * k * t) / k ** 1.3 for k in range(1, harmonics + 1))
    attack = np.minimum(t / 0.01, 1)
    return wave * attack * np.exp(-t * decay)


def place(buf, sample, at):
    start = int(at * SR)
    end = min(start + len(sample), len(buf))
    if start < len(buf):
        buf[start:end] += sample[:end - start]


def kick():
    t = np.arange(int(0.25 * SR)) / SR
    return np.sin(2 * np.pi * (50 * t + 60 * (1 - np.exp(-t * 30)) / 30)) * np.exp(-t * 14)


def snare():
    t = np.arange(int(0.2 * SR)) / SR
    return (rng.normal(0, .5, len(t)) + np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 22)


def hat():
    t = np.arange(int(0.05 * SR)) / SR
    noise = rng.normal(0, .4, len(t))
    return np.diff(noise, prepend=0) * np.exp(-t * 90)


def vocal(detune_cents=0):
    """A sung line with vibrato and deliberately sharp and flat notes."""
    melody = [(0, 69, 25), (2, 72, -30), (4, 71, 35), (6, 69, -20), (8, 67, 30), (10, 69, -35), (12, 72, 20), (14, 76, -25)]
    out = np.zeros(LENGTH)
    for bar in range(0, BARS, 4):
        for beat, midi, cents in melody:
            freq = 440 * 2 ** ((midi - 69 + (cents + detune_cents) / 100) / 12)
            dur = BEAT * 1.8
            t = np.arange(int(dur * SR)) / SR
            vib = 0.35 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(t / 0.3, 1)
            phase = 2 * np.pi * np.cumsum(freq * 2 ** (vib / 12)) / SR
            wave = sum(np.sin(phase * k) * 1 / k ** 1.1 for k in range(1, 12))
            env = np.minimum(t / 0.03, 1) * np.minimum((dur - t) / 0.08, 1)
            place(out, wave * env, (bar * 4 + beat) * BEAT)
    return out / np.abs(out).max() * .7


def main(folder):
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    tracks = {name: np.zeros(LENGTH) for name in ("Kick", "Snare", "HiHat", "Bass", "Piano")}
    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
    for bar in range(BARS):
        at = bar * 4 * BEAT
        chord = chords[bar % 4]
        for b in range(4):
            place(tracks["Kick"], kick(), at + b * BEAT if b % 2 == 0 else at + (b + .5) * BEAT)
            if b % 2:
                place(tracks["Snare"], snare(), at + b * BEAT)
        for i in range(8):
            place(tracks["HiHat"], hat(), at + i * BEAT / 2)
        place(tracks["Bass"], note(440 * 2 ** ((chord[0] - 24 - 69) / 12), BEAT * 3.8, 3, .4), at)
        for midi in chord:
            place(tracks["Piano"], note(440 * 2 ** ((midi + 12 - 69) / 12), BEAT * 3.8, 5, .9) * .4, at)
    tracks["Lead Vocal"] = vocal()
    tracks["Vocal Harmony L"] = vocal(12) * .5
    tracks["Texture 07"] = rng.normal(0, 0.02, LENGTH)
    for name, audio in tracks.items():
        peak = np.abs(audio).max() or 1
        sf.write(root / f"{name}.wav", audio / peak * .8, SR, subtype="PCM_24")
    return root


if __name__ == "__main__":
    print(main(sys.argv[1]))
