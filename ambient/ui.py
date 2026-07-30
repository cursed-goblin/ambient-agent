"""
Local web UI -- HTTP server, SSE event stream, command endpoint, config API.

New in this version:
  GET  /api/config  -- return current provider config (key masked)
  POST /api/config  -- save new config, fire on_config_change callback

The UI now handles its own setup wizard and settings panel.
No terminal wizard is needed on first run.
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
    Serves the page, streams state, accepts typed commands, handles config.

    Callbacks (all optional, all called on an HTTP thread -- must be quick):
      on_command(text)       -- a command was typed or spoken in the UI
      get_config()           -- return the current provider cfg dict
      on_config_change(cfg)  -- user saved new settings; reload the escalator
    """

    def __init__(
        self,
        on_command: Callable[[str], None],
        get_config: Optional[Callable[[], dict]] = None,
        on_config_change: Optional[Callable[[dict], None]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        self.on_command = on_command
        self.get_config = get_config
        self.on_config_change = on_config_change
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

    def notify(self, kind: str, payload: Optional[dict] = None) -> None:
        """Generic push for setup-complete, config-saved, etc."""
        self._broadcast({"type": kind, **(payload or {})})

    # -- lifecycle ------------------------------------------------------
    def start(self) -> str:
        server = ThreadingHTTPServer((self.host, self.port), _make_handler(self))
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
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

    def _config_public(self) -> dict:
        """Return current config with key masked for the browser."""
        cfg = {}
        if self.get_config:
            try:
                cfg = dict(self.get_config() or {})
            except Exception:  # noqa: BLE001
                cfg = {}
        if cfg.get("api_key"):
            cfg["api_key"] = "*" * min(8, len(cfg["api_key"]))
            cfg["key_set"] = True
        else:
            cfg["key_set"] = False
        return cfg


def _make_handler(ui: UiServer):  # noqa: C901
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:
            pass

        # -- GET --------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._serve_index()
            elif path == "/events":
                self._serve_events()
            elif path == "/state":
                self._send_json(current().snapshot())
            elif path == "/api/config":
                self._send_json(ui._config_public())
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
                        chunk = ": ping\n\n"
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                ui._unregister(q)

        # -- POST -------------------------------------------------------
        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/command":
                self._handle_command()
            elif path == "/api/config":
                self._handle_config_save()
            else:
                self.send_error(404)

        def _handle_command(self) -> None:
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
                except Exception as exc:  # noqa: BLE001
                    log_event("ui_command_error", error=str(exc)[:200])
            self._send_json({"ok": True})

        def _handle_config_save(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except ValueError:
                self.send_error(400, "bad json")
                return

            import ambient.provider as prov

            provider_name = body.get("provider", "none")
            preset = dict(prov.PRESETS.get(provider_name, prov.PRESETS["none"]))

            # Overlay the user-supplied fields.
            if body.get("api_key") and body["api_key"] not in ("", "*" * 8):
                preset["api_key"] = body["api_key"].strip()
            if body.get("base_url"):
                preset["base_url"] = body["base_url"].strip().rstrip("/")
            if body.get("model"):
                preset["model"] = body["model"].strip()

            # For edits that don't resupply the key, keep the old one.
            if provider_name == "groq" and not preset.get("api_key"):
                old_cfg = ui.get_config() if ui.get_config else None
                if old_cfg and old_cfg.get("provider") == "groq":
                    preset["api_key"] = old_cfg.get("api_key", "")

            prov.save(preset)
            log_event("ui_config_saved", provider=provider_name)

            if ui.on_config_change:
                try:
                    ui.on_config_change(preset)
                except Exception as exc:  # noqa: BLE001
                    log_event("ui_config_reload_error", error=str(exc)[:200])

            # Broadcast so all open tabs update their badges.
            ui.notify("config_saved", {"provider": prov.describe(preset),
                                        "is_setup": False})
            self._send_json({"ok": True, "provider": prov.describe(preset)})

        def _send_json(self, payload: dict) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler
