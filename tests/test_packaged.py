"""Run the packaged exe with a bare PATH, as on a PC with no Python, ffmpeg or Node."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "dist" / "StudioAutopilot" / "StudioAutopilot.exe"
PORT = 8791
BASE = f"http://127.0.0.1:{PORT}"


def main():
    stems = Path(sys.argv[1]) / "stems"
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PATH", "PYTHONPATH")}
    env["PATH"] = r"C:\Windows\System32"
    proc = subprocess.Popen([str(EXE), "--serve", f"--port={PORT}"], env=env, cwd=str(EXE.parent))
    try:
        for _ in range(120):
            try:
                page = requests.get(BASE, timeout=2).text
                break
            except requests.RequestException:
                time.sleep(1)
        token = re.search(r'TOKEN = "(.+?)"', page).group(1)
        head = {"X-Token": token}
        assert requests.get(BASE + "/api/status", timeout=10).status_code == 403
        print(requests.get(BASE + "/api/status", headers=head, timeout=30).json())

        def run(path, data, wait=400):
            job = requests.post(BASE + "/api/" + path, json=data, headers=head, timeout=30).json()["job"]
            end = time.time() + wait
            while time.time() < end:
                info = next(j for j in requests.get(BASE + "/api/jobs", headers=head, timeout=10).json() if j["id"] == job)
                if info["status"] in ("done", "error", "cancelled"):
                    print(path, info["status"], info["error"] or "", round(info["elapsed"]), "s")
                    return info
                time.sleep(1)
            raise TimeoutError(path)

        scan = requests.post(BASE + "/api/mix/scan", json={"folder": str(stems)}, headers=head, timeout=60).json()
        assert run("mix/run", {"stems": scan["stems"], "name": "Pack", "variants": False})["status"] == "done"
        assert run("tune/run", {"vocals": [str(stems / "Lead Vocal.wav")], "key": "A", "scale": "Minor"})["status"] == "done"
        assert run("download", {"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw", "kind": "mp3"})["status"] == "done"
        speech = Path(sys.argv[1]) / "speech.wav"
        result = run("lyrics/extract", {"audio": str(speech), "language": "en", "model": "tiny"})
        assert result["status"] == "done"
        print(json.dumps(result["result"]["lines"][0]["text"]))
    finally:
        proc.kill()
    print("OK")


if __name__ == "__main__":
    main()
