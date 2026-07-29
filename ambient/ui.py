"""
Simple local UI (interim Phase 2).

A tiny stdlib HTTP server plus Server-Sent Events. No Tauri, no Electron, no
npm, no build step -- open a browser at http://127.0.0.1:8765 and you can see
what the agent is doing and type to it.

Why a browser page rather than the planned transparent overlay: it works today
on a laptop with no GPU, and it works over a forwarded port from a cloud dev
box, which is where this is being developed. The real always-on-top overlay is
still Phase 2 proper. This exists so the agent is observable in the meantime.

Binds to 127.0.0.1 by default. It is not authenticated, so do not expose it on
a network you do not trust.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

import config
from ambient.state import current, log_event

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
INDEX = UI_DIR / "index.html"


class UiServer:
    """
    Serves the page, streams state, accepts typed commands.

    on_command is called on the HTTP thread, so the callback must be quick and
    thread-safe. main.py hands work to the assistant's own queue.
    """

    def __init__(self, on_command: Callable[[str], None],
                 host: Optional[str] = None,
                 port: Optional[int] = None) -> None:
        self.on_command = on_command
        self.host = host or config.UI_HOST
        self.port = int(port or config.UI_PORT)
        self._clients: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- outward events -------------------------------------------------
    def _broadcast(self, payload: dict) -> None:
        data = json.dumps(payload, default=str)
        with self._lock:
            dead = []
            for q in self._clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._clients.remove(q)

    def push_state(self) -> None:
        self._broadcast({"type": "state", "state": current().snapshot()})

    def add_message(self, role: str, text: str) -> None:
        """role: 'you' | 'agent' | 'system'"""
        entry = {"role": role, "text": text, "ts": time.time()}
        with self._lock:
            self._history.append(entry)
            del self._history[:-200]
        self._broadcast({"type": "message", "message": entry})

    def set_banner(self, text: str) -> None:
        self._broadcast({"type": "banner", "text": text})

    # -- lifecycle ------------------------------------------------------
    def start(self) -> str:
        server = ThreadingHTTPServer((self.host, self.port), _make_handler(self))
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

        # Mirror every state change into the browser.
        current().subscribe(lambda _s: self.push_state())

        url = "http://" + str(self.host) + ":" + str(self.port)
        log_event("ui_started", url=url)
        return url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # -- used by the handler -------------------------------------------
    def _register(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=100)
        with self._lock:
            self._clients.append(q)
            history = list(self._history)
        q.put_nowait(json.dumps({"type": "state", "state": current().snapshot()}))
        for entry in history[-40:]:
            q.put_nowait(json.dumps({"type": "message", "message": entry}))
        return q

    def _unregister(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)


def _make_handler(ui: UiServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:
            pass  # keep the agent's console readable

        # -- GET --------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._serve_index()
            elif self.path == "/events":
                self._serve_events()
            elif self.path == "/state":
                self._send_json(current().snapshot())
            else:
                self.send_error(404)

        def _serve_index(self) -> None:
            try:
                body = INDEX.read_bytes()
            except OSError:
                self.send_error(500, "ui/index.html is missing")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = ui._register()
            try:
                while True:
                    try:
                        data = q.get(timeout=15)
                        chunk = f"data: {data}\n\n"
                    except queue.Empty:
                        chunk = ": ping\n\n"      # keep the connection warm
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                ui._unregister(q)

        # -- POST -------------------------------------------------------
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/command":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                text = (json.loads(raw.decode("utf-8")).get("text") or "").strip()
            except ValueError:
                self.send_error(400, "bad json")
                return
            if text:
                log_event("ui_command", text=text[:200])
                try:
                    ui.on_command(text)
                except Exception as exc:  # a UI must never crash the agent
                    log_event("ui_command_error", error=str(exc)[:200])
            self._send_json({"ok": True})

        def _send_json(self, payload: dict) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler
