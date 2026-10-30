"""End-to-end checks for the audio engines on a synthetic song."""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


class FakeJob:
    def __init__(self):
        self.last = None

    def update(self, progress=None, stage=None):
        if stage and stage != self.last:
            self.last = stage

    def check(self):
        pass


def main():
    import make_test_song
    from studio import analysis, mixer, tuner

    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    stems_dir = make_test_song.main(work / "stems")

    start = time.time()
    facts = analysis.analyze_file(stems_dir / "Kick.wav", want_beats=True)
    print("kick bpm", facts["bpm"], "beats", len(facts["beats"]), f"{time.time() - start:.1f}s")
    assert abs(facts["bpm"] - 100) < 2 or abs(facts["bpm"] - 200) < 3 or abs(facts["bpm"] - 50) < 2, facts["bpm"]

    stems = mixer.scan(stems_dir)
    print({s["name"]: s["role"] for s in stems})
    roles = {s["name"]: s["role"] for s in stems}
    assert roles["Lead Vocal"] == "lead_vocal" and roles["Kick"] == "kick" and roles["Vocal Harmony L"] == "harmony"

    start = time.time()
    result = mixer.mix_song(FakeJob(), stems, "Test", "bollywood")
    print("mix", result, f"{time.time() - start:.1f}s")
    audio, sr = sf.read(result["master"])
    lufs = pyln.Meter(sr).integrated_loudness(audio)
    print("measured lufs", round(lufs, 2), "peak", round(20 * np.log10(np.abs(audio).max()), 2))
    assert abs(lufs + 11) < 1.0 and np.abs(audio).max() < 1.0
    assert result["true_peak"] <= -0.99, result["true_peak"]
    folder = Path(result["folder"])
    wavs = sorted((folder / "Stems_for_Ableton").glob("*.wav"))
    premaster, _ = sf.read(next(folder.glob("*premaster.wav")))
    total = sum(sf.read(w)[0] for w in wavs)
    residual = 20 * np.log10(np.sqrt(np.mean((total - premaster) ** 2)) + 1e-12)
    print("stems null test dB", round(residual, 1), "files", len(wavs))
    assert residual < -60, "Ableton stems must sum to the premaster"

    start = time.time()
    tuned = tuner.tune(FakeJob(), [str(stems_dir / "Lead Vocal.wav")], music=str(stems_dir / "Piano.wav"), key="A",
                       scale="Minor", preset="tight")
    print("tune", {k: v for k, v in tuned.items() if k != "chart"}, f"{time.time() - start:.1f}s")
    summary = tuned["files"][0]
    assert summary["notes"] >= 8 and summary["avg_error_after"] < summary["avg_error_before"] * 0.4
    print("OK")


if __name__ == "__main__":
    main()
