"""Studio Autopilot launcher: runs the local server and opens the app window."""
import faulthandler
import multiprocessing
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def browser_path():
    roots = [os.environ.get(k, "") for k in ("ProgramFiles(x86)", "ProgramFiles", "LocalAppData")]
    apps = [r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"]
    for app in apps:
        for root in roots:
            path = os.path.join(root, app)
            if root and os.path.exists(path):
                return path
    return None


def open_window(url):
    """Edge and Chrome app mode gives a clean window with no tabs or address bar."""
    from studio.core import DATA_DIR

    path = browser_path()
    if path and "--browser" not in sys.argv:
        subprocess.Popen([path, f"--app={url}", f"--user-data-dir={DATA_DIR / 'window'}", "--no-first-run",
                          "--no-default-browser-check", "--window-size=1380,900"])
    else:
        webbrowser.open(url)


def main():
    from waitress import serve

    from studio import server

    fixed = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--port=")), None)
    port = int(fixed) if fixed else free_port()
    threading.Thread(target=serve, args=(server.app,), daemon=True,
                     kwargs={"host": "127.0.0.1", "port": port, "threads": 8}).start()
    if "--serve" in sys.argv:
        while True:
            time.sleep(3600)
    open_window(f"http://127.0.0.1:{port}/")
    # The page polls constantly, so a long silence with nothing running means the window was closed.
    while True:
        time.sleep(10)
        if server.idle_seconds() > 120:
            os._exit(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # tuner workers need this inside the packaged exe
    if getattr(sys, "frozen", False):
        # A windowed exe has no console, so errors and native crashes go to a log file.
        from studio.core import DATA_DIR
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log = open(DATA_DIR / "app.log", "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
        faulthandler.enable(log)
    main()
