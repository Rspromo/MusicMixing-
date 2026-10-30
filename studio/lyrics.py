"""Lyric extraction with a local Whisper model or the OpenAI API."""
import json
import tempfile
from pathlib import Path

import requests

from .analysis import decode_mono
from .core import MODELS_DIR, load_settings, media_info, run, tool

API = "https://api.openai.com/v1"
_cache = {}


def _model(size):
    from faster_whisper import WhisperModel

    if size in _cache:
        return _cache[size]
    try:
        import ctranslate2
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        device = "cpu"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    kind = "float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(size, device=device, compute_type=kind, download_root=str(MODELS_DIR))
    except Exception:
        model = WhisperModel(size, device="cpu", compute_type="int8", download_root=str(MODELS_DIR))
    _cache.clear()  # one model at a time keeps RAM use low
    _cache[size] = model
    return model


def _local(job, audio, language, size):
    job.update(0.03, f"Whisper model '{size}' load ho raha hai (pehli baar download hoga)")
    model = _model(size)
    samples = decode_mono(audio, 16000)  # ffmpeg reads every format Whisper's own decoder might not
    pieces, info = model.transcribe(samples, language=language or None, word_timestamps=True, beam_size=5,
                                    vad_filter=True, vad_parameters={"threshold": 0.3, "min_silence_duration_ms": 800},
                                    condition_on_previous_text=False)
    lines = []
    for piece in pieces:
        job.update(0.08 + 0.8 * min(piece.end / max(info.duration, 1), 1), f"Lyrics sun raha hai: {piece.end:.0f}s")
        words = [{"start": w.start, "end": w.end, "word": w.word.strip()} for w in (piece.words or [])]
        lines.append({"start": piece.start, "end": piece.end, "text": piece.text.strip(), "words": words})
    return lines, info.language


def _openai(job, audio, language, key):
    job.update(0.05, "Audio ko chhota kar raha hai")
    with tempfile.TemporaryDirectory() as temp:
        small = Path(temp) / "audio.mp3"
        run([tool("ffmpeg"), "-y", "-v", "error", "-i", str(audio), "-ac", "1", "-b:a", "64k", str(small)])
        if small.stat().st_size > 24_000_000:
            raise RuntimeError("Audio OpenAI ki 25 MB limit se bada hai. Local Whisper use karein.")
        job.update(0.2, "OpenAI se lyrics nikal raha hai")
        fields = [("model", "whisper-1"), ("response_format", "verbose_json"),
                  ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")]
        if language:
            fields.append(("language", language))
        with small.open("rb") as handle:
            reply = requests.post(f"{API}/audio/transcriptions", headers={"Authorization": f"Bearer {key}"},
                                  data=fields, files={"file": ("audio.mp3", handle, "audio/mpeg")}, timeout=600)
    if reply.status_code != 200:
        raise RuntimeError(f"OpenAI error {reply.status_code}: {reply.text[:300]}")
    data = reply.json()
    words = [{"start": w["start"], "end": w["end"], "word": w["word"].strip()} for w in data.get("words", [])]
    lines = []
    for seg in data.get("segments", []):
        inside = [w for w in words if seg["start"] - 0.05 <= w["start"] < seg["end"]]
        lines.append({"start": seg["start"], "end": seg["end"], "text": seg["text"].strip(), "words": inside})
    return lines, data.get("language", language)


def split_lines(lines, max_words=8):
    result = []
    for line in lines:
        words = line.get("words") or []
        if len(words) <= max_words:
            result.append(line)
            continue
        size = max_words if len(words) % max_words != 1 else max_words - 1
        for i in range(0, len(words), size):
            part = words[i:i + size]
            result.append({"start": part[0]["start"], "end": part[-1]["end"],
                           "text": " ".join(w["word"] for w in part), "words": part})
    return result


def chat_json(prompt, key, model):
    reply = requests.post(f"{API}/chat/completions", headers={"Authorization": f"Bearer {key}"}, timeout=180, json={
        "model": model, "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}]})
    if reply.status_code != 200:
        raise RuntimeError(f"OpenAI error {reply.status_code}: {reply.text[:300]}")
    return json.loads(reply.json()["choices"][0]["message"]["content"])


def polish(lines, roman, key, model):
    """Fix misheard words, optionally in Roman script, without changing the line count."""
    texts = [l["text"] for l in lines]
    style = "Write in Roman script (Hinglish)." if roman else "Keep the original script."
    data = chat_json("These are song lyric lines from speech recognition. Fix obvious mishearings and spelling. "
                     f"{style} Return JSON {{\"lines\": [...]}} with exactly {len(texts)} strings in the same order.\n"
                     + json.dumps(texts, ensure_ascii=False), key, model)
    fixed = data.get("lines", [])
    if len(fixed) != len(lines):
        return lines
    for line, text in zip(lines, fixed):
        if text.strip() != line["text"]:
            words = text.split()
            old = line.get("words") or []
            line["text"] = text.strip()
            if len(words) == len(old):
                for w, new in zip(old, words):
                    w["word"] = new
            else:
                span = (line["end"] - line["start"]) / max(len(words), 1)
                line["words"] = [{"start": line["start"] + i * span, "end": line["start"] + (i + 1) * span, "word": w}
                                 for i, w in enumerate(words)]
    return lines


def keywords(lines, title, key, model):
    text = " / ".join(l["text"] for l in lines)[:3000]
    data = chat_json(f"Song title: {title}. Lyrics: {text}\nSuggest 8 simple English stock-footage search terms "
                     "matching the mood and imagery. Return JSON {\"terms\": [...]}.", key, model)
    return [t for t in data.get("terms", []) if isinstance(t, str)][:8]


def extract(job, audio, language="", engine="local", model=None, polish_text=False, roman=False, title=""):
    settings = load_settings()
    if engine == "openai":
        if not settings["openai_key"]:
            raise RuntimeError("Settings mein OpenAI key daalein ya Local Whisper chunein.")
        lines, found = _openai(job, audio, language, settings["openai_key"])
    else:
        lines, found = _local(job, audio, language, model or settings["whisper_model"])
    lines = split_lines(lines)
    terms = []
    if settings["openai_key"] and lines:
        if polish_text or roman:
            job.update(0.9, "Lyrics saaf kar raha hai")
            lines = polish(lines, roman, settings["openai_key"], settings["openai_chat_model"])
        job.update(0.95, "Video ke liye keywords soch raha hai")
        terms = keywords(lines, title, settings["openai_key"], settings["openai_chat_model"])
    return {"lines": lines, "language": found, "terms": terms, "duration": media_info(audio)["duration"]}


def stamp(t, comma=False):
    h, rem = divmod(max(t, 0), 3600)
    m, s = divmod(rem, 60)
    if comma:
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")
    return f"{int(m + h * 60):02d}:{s:05.2f}"


def save_text(lines, folder, name):
    lrc = "\n".join(f"[{stamp(l['start'])}]{l['text']}" for l in lines)
    srt = "\n\n".join(f"{i}\n{stamp(l['start'], True)} --> {stamp(l['end'], True)}\n{l['text']}"
                      for i, l in enumerate(lines, start=1))
    (Path(folder) / f"{name}.lrc").write_text(lrc, encoding="utf-8")
    (Path(folder) / f"{name}.srt").write_text(srt, encoding="utf-8")
    (Path(folder) / f"{name}_lyrics.txt").write_text("\n".join(l["text"] for l in lines), encoding="utf-8")
