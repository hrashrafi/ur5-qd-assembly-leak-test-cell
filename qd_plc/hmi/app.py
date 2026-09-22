"""HTTP front end for the operator panel.

Deliberately stdlib-only. The panel is a thin window onto Modbus registers -
one JSON endpoint and a static page - and a web framework would add a
dependency and a deployment story for no benefit.

    GET  /                 the panel
    GET  /api/state        everything the page renders, one object
    POST /api/cmd          {"tag": "HMI_START", "value": 1}
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .client import PlcClient
from .tagmap import NotWritable

STATIC = Path(__file__).resolve().parent / "static"


class Handler(BaseHTTPRequestHandler):
    plc: PlcClient = None            # set by serve()
    protocol_version = "HTTP/1.1"

    # The default handler logs every request to stderr, which at 4 polls a
    # second buries anything worth reading.
    def log_message(self, fmt, *args):
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            self._json(200, self.plc.snapshot())
            return
        if path == "/":
            path = "/index.html"
        target = (STATIC / path.lstrip("/")).resolve()
        # Keep path traversal out: serve only what is inside the static dir.
        if not str(target).startswith(str(STATIC)) or not target.is_file():
            self._json(404, {"error": "not found"})
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), ctype)

    def do_POST(self) -> None:
        if self.path != "/api/cmd":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            tag, value = body["tag"], body["value"]
        except (ValueError, KeyError):
            self._json(400, {"error": "expected {tag, value}"})
            return
        try:
            self.plc.command(tag, value)
        except NotWritable as exc:
            # The panel owns a documented range and nothing else; refusing
            # here keeps a UI bug from turning into a permit race with the
            # supervisor that would look exactly like a PLC fault.
            self._json(403, {"error": str(exc)})
            return
        except KeyError:
            self._json(404, {"error": f"no tag {tag!r}"})
            return
        except Exception as exc:                      # noqa: BLE001
            self._json(502, {"error": f"write failed: {exc}"})
            return
        self._json(200, {"ok": True, "tag": tag, "value": value})


def serve(plc_host: str, plc_port: int, http_port: int,
          interval: float = 0.25) -> None:
    plc = PlcClient(plc_host, plc_port, interval=interval)
    plc.start()
    Handler.plc = plc
    server = ThreadingHTTPServer(("127.0.0.1", http_port), Handler)
    print(f"  operator panel  http://127.0.0.1:{http_port}")
    print(f"  polling PLC at  {plc_host}:{plc_port} every "
          f"{int(interval * 1000)}ms\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
        plc.stop()
