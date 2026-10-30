"""Checks for tuned pitch, lyric video, video editor, Premiere XML and Ableton reading."""
import glob
import gzip
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pyworld
import soundfile as sf

from studio import ableton, lyricvideo, videoeditor
from studio.core import media_info, run, tool
from test_engines import FakeJob


def pitch_error(path, sr_scale=(0, 2, 3, 5, 7, 8, 10), root=9):
    audio, sr = sf.read(path)
    audio = audio.mean(axis=1) if audio.ndim > 1 else audio
    f0, _ = pyworld.dio(audio[:sr * 20].astype(np.float64), sr, frame_period=5)
    midi = 69 + 12 * np.log2(f0[f0 > 0] / 440)
    allowed = np.array([(root + s) % 12 for s in sr_scale])
    # A 3-point median removes vibrato so that the note centre is what gets judged.
    size = 100
    centers = [np.median(midi[i:i + size]) for i in range(0, len(midi) - size, size)]
    errors = []
    for c in centers:
        delta = (c - allowed[:, None] + 6) % 12 - 6
        errors.append(np.abs(delta).min())
    return float(np.mean(errors) * 100)


def main():
    work = Path(sys.argv[1])
    stems = work / "stems"
    song = glob.glob(str(ROOT / "StudioData/Output/Mixes/Test_*/Test_MASTER.wav"))[-1]
    tuned = glob.glob(str(ROOT / "StudioData/Output/Tuned/Lead Vocal_*/Lead Vocal_TUNED.wav"))[-1]
    print("measured cents off-scale: raw", round(pitch_error(stems / "Lead Vocal.wav"), 1), "tuned", round(pitch_error(tuned), 1))

    clips = work / "clips"
    clips.mkdir(exist_ok=True)
    ff = tool("ffmpeg")
    # Two clips carry the song as scratch audio at known positions, the others are silent b-roll.
    run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=30", "-ss", "4", "-i", song, "-t", "30",
         "-map", "0:v", "-map", "1:a", "-shortest", "-c:v", "libx264", "-preset", "ultrafast", str(clips / "A_perf.mp4")])
    run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "smptebars=s=1920x1080:r=30:d=24", "-ss", "20", "-i", song, "-t", "24",
         "-map", "0:v", "-map", "1:a", "-shortest", "-c:v", "libx264", "-preset", "ultrafast", str(clips / "B_perf.mp4")])
    for i, source in enumerate(("mandelbrot=s=1280x720:r=30", "rgbtestsrc=s=1280x720:r=30", "cellauto=s=1280x720:r=30")):
        run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", source, "-t", "15", "-c:v", "libx264", "-preset", "ultrafast",
             str(clips / f"broll{i}.mp4")])
    names = sorted(str(p) for p in clips.iterdir())

    for style in ("performance", "cinematic", "reel"):
        result = videoeditor.edit(FakeJob(), song, names, style, length=20)
        info = media_info(result["file"])
        print(style, {k: result[k] for k in ("cuts", "bpm", "clips_used", "synced")}, info["width"], info["height"], round(info["duration"], 2))
        assert abs(info["duration"] - 20) < 0.5 and info["has_audio"] and result["synced"] == 2
        for xml in Path(result["premiere"]).glob("*.xml"):
            tree = ET.parse(xml)
            items = tree.findall(".//video/track/clipitem")
            assert len(items) == result["cuts"], xml
            ids = [f.get("id") for f in tree.iter("file") if f.find("pathurl") is not None]
            assert len(ids) == len(set(ids)), "each file defined once"
        report = Path(result["folder"]) / "Edit_Report.json"
        print(report.read_text())

    lines = [{"start": 1.0, "end": 4.0, "text": "तेरी यादों में खोया हूँ", "words": [
        {"start": 1.0, "end": 1.6, "word": "तेरी"}, {"start": 1.6, "end": 2.3, "word": "यादों"},
        {"start": 2.3, "end": 2.6, "word": "में"}, {"start": 2.6, "end": 3.2, "word": "खोया"}, {"start": 3.2, "end": 4.0, "word": "हूँ"}]},
        {"start": 4.5, "end": 8.0, "text": "Roman line for testing", "words": []}]
    images = []
    for i, color in enumerate(("0x334466", "0x664433")):
        path = work / f"bg{i}.png"
        run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:s=1920x1080:d=0.1", "-frames:v", "1", str(path)])
        images.append(str(path))
    for style, aspect in (("karaoke", "16:9"), ("word_pop", "9:16"), ("next_line", "1:1")):
        out = lyricvideo.render(FakeJob(), song, lines, [], images, {
            "style": style, "aspect": aspect, "source": "local", "preview": 10, "height": 720, "title": "Test Song", "artist": "Studio",
            "visualizer": True, "clip_length": 4})
        info = media_info(out["file"])
        print(style, info["width"], info["height"], round(info["duration"], 2))
        assert abs(info["duration"] - 10) < 0.5
    run([ff, "-y", "-v", "error", "-ss", "2.5", "-i", out["file"], "-frames:v", "1", str(work / "frame.png")])
    print("online", len(lyricvideo.commons_search("sunset")), "commons results")

    als = work / "t.als"
    xml = f"""<Ableton><LiveSet><Tracks><AudioTrack><Name><EffectiveName Value="Lead Vox"/></Name>
    <AudioClip Time="0"><CurrentStart Value="8"/><CurrentEnd Value="16"/><Loop><LoopStart Value="0"/></Loop><IsWarped Value="false"/>
    <SampleRef><FileRef><RelativePath Value="stems/Lead Vocal.wav"/><Path Value="C:/nowhere/Lead Vocal.wav"/></FileRef></SampleRef></AudioClip>
    </AudioTrack></Tracks><MainTrack><DeviceChain><Mixer><Tempo><Manual Value="100"/></Tempo></Mixer></DeviceChain></MainTrack></LiveSet></Ableton>"""
    with gzip.open(als, "wb") as handle:
        handle.write(xml.encode())
    bounced = ableton.bounce(FakeJob(), als)
    audio, sr = sf.read(bounced["files"][0])
    first = np.argmax(np.abs(audio).max(axis=1) > 0.01) / sr
    print("ableton first sound at", round(first, 2), "expected 4.8")
    assert abs(first - 4.8) < 0.05
    print("OK")


if __name__ == "__main__":
    main()
