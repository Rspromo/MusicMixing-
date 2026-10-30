"""Online checks: Spotify name lookup and local Whisper on generated speech."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from studio import downloader, lyrics
from test_engines import FakeJob

POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def main():
    work = Path(sys.argv[1])
    print("spotify", downloader.spotify_to_search("https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT"))
    speech = work / "speech.wav"
    script = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.SetOutputToWaveFile('{speech}'); $s.Speak('Hello my friend. This is a test of the lyric extraction system. "
              "The music plays all night long.'); $s.Dispose()")
    subprocess.run([POWERSHELL, "-NoProfile", "-Command", script], check=True)
    result = lyrics.extract(FakeJob(), str(speech), "en", "local", "tiny")
    print([(round(l["start"], 1), l["text"], len(l["words"])) for l in result["lines"]])
    assert result["lines"] and "test" in " ".join(l["text"] for l in result["lines"]).lower()
    print("OK")


if __name__ == "__main__":
    main()
