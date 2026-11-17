"""
Lanzador Unificado de la Aplicación Gráfica DataVoley Studio (run_app.py)

Inicia automáticamente:
1. El motor táctico FIVB (R1-R6) y de homografía 2D con datos iniciales cargados.
2. El servidor backend FastAPI en http://127.0.0.1:8055 esperando confirmación (/health).
3. Abre la ventana gráfica nativa dedicada (--app) con la pista 2D interactiva y gráficos en tiempo real.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn
from data_voley.api.server import app


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def _wait_until_server_ready(url: str, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    health_url = f"{url}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.15)
    return False


def _start_api_background(host: str, port: int) -> threading.Thread:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="datavoley-api-thread", daemon=True)
    thread.start()
    return thread


def _find_native_app_browser() -> str | None:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return str(path)
    return None


def _open_ui_window(url: str) -> subprocess.Popen | None:
    browser_bin = _find_native_app_browser()
    if browser_bin:
        return subprocess.Popen(
            [
                browser_bin,
                f"--app={url}",
                "--window-size=1480,920",
            ],
            shell=False,
        )
    webbrowser.open(url)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Lanzador Visual DataVoley Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host local")
    parser.add_argument("--port", type=int, default=8055, help="Puerto local")
    parser.add_argument("--no-browser", action="store_true", help="Solo iniciar servidor sin abrir ventana")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    if not _is_port_in_use(args.host, args.port):
        _start_api_background(args.host, args.port)
        _wait_until_server_ready(url)

    if not args.no_browser:
        _open_ui_window(url)

    print(f"[DataVoley Studio] Interfaz gráfica activa en: {url}")
    print("Presiona Ctrl+C para cerrar el servidor.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
