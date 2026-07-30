"""
A real window for the local UI (spec 4.13, interim).

The end state is a Tauri transparent overlay. Until then, this wraps the
existing loopback web UI in an actual window so the assistant stops being a
terminal program.

Three strategies, best first:

1. pywebview -- a genuine native window (GTK + WebKit on Linux). No address
   bar, no tabs, no browser chrome. `pip install pywebview`.
2. The default browser, opened at the right URL. Works everywhere, including
   Android, where there is no desktop toolkit at all.
3. Print the URL and let the user click it.

Why a subprocess: pywebview insists on owning the main thread, and the main
thread here belongs to the assistant loop. Rather than invert the whole
program around a GUI toolkit, the window runs as its own short process --
`python3 -m ambient.window <url>` -- which also means a crashed or closed
window can never take the assistant down with it.

Nothing in here is ever fatal. If every strategy fails the assistant keeps
running headless and says so.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

TITLE = "Ambient"
WIDTH = 480
HEIGHT = 720


def _has_pywebview() -> bool:
    try:
        import webview  # noqa: F401
    except Exception:
        return False
    return True


def _has_display() -> bool:
    """No point starting a toolkit with nothing to draw on."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def open_window(url: str, title: str = TITLE) -> Optional[subprocess.Popen]:
    """
    Show `url` in a window. Returns the window process if one was spawned,
    else None. Callers should treat None as "fine, carry on".
    """
    if _has_pywebview() and _has_display():
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "ambient.window", url, title],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("[window] Native window opened.")
            return proc
        except Exception as exc:  # noqa: BLE001
            print("[window] Native window failed (" + str(exc)[:120] + ").")

    if _try_browser(url):
        return None

    print("[window] No window available. Open this yourself: " + url)
    if not _has_pywebview():
        print("[window] For a proper window: pip install pywebview")
    return None


def _try_browser(url: str) -> bool:
    # Android/Termux: no DISPLAY, no webbrowser backend, but this hands the
    # URL to the phone's actual browser.
    for tool in ("termux-open-url", "xdg-open"):
        try:
            subprocess.Popen([tool, url],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            print("[window] Opened in your browser (" + tool + ").")
            return True
        except FileNotFoundError:
            continue
        except Exception:  # noqa: BLE001
            continue

    try:
        import webbrowser
        if webbrowser.open(url):
            print("[window] Opened in your default browser.")
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def close_window(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _run_webview(url: str, title: str) -> int:
    """Child-process entry point. Blocks until the window is closed."""
    import webview

    webview.create_window(
        title,
        url,
        width=WIDTH,
        height=HEIGHT,
        resizable=True,
        text_select=True,
    )
    webview.start()
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    name = sys.argv[2] if len(sys.argv) > 2 else TITLE
    raise SystemExit(_run_webview(target, name))
