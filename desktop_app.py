"""Energy Management System : fenêtre d'application (sans navigateur).

Lance l'application Streamlit en arrière-plan et l'affiche dans une fenêtre
native (pywebview). Fermer la fenêtre arrête aussi le serveur.

Prérequis (une seule fois) :  python -m pip install pywebview
Lancement                   :  python desktop_app.py   (ou double-clic sur Lancer_Fenetre.bat)
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parent
TITLE = "Energy Management System"
LOADING = """
<body style="font-family:Segoe UI,Arial,sans-serif;background:#fcfcfb;color:#0b0b0b;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
<div style="text-align:center"><div style="font-size:42px">&#9889;</div>
<h2>Energy Management System</h2><p style="color:#52514e">Chargement de l'application...</p></div>
</body>
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, timeout: int = 90) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    port = free_port()
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    server = subprocess.Popen(cmd, cwd=ROOT, creationflags=flags,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    window = webview.create_window(TITLE, html=LOADING, width=1320, height=860,
                                   min_size=(980, 640))

    def boot():
        if wait_for_server(base + "/_stcore/health"):
            window.load_url(base)
        else:
            window.load_html("<h3 style='font-family:sans-serif'>Impossible de démarrer l'application.<br>"
                             "Lance 'python -m streamlit run app.py' pour voir l'erreur.</h3>")

    try:
        webview.start(boot)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
