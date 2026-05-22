from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from hemodynamic_graph import build_hemodynamic_graph
from hemodynamic_predictor import (
    DEFAULT_DATA_PATH,
    DEFAULT_SUMMARY_INFO_PATH,
    HemodynamicPredictor,
)


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"


class HemodynamicServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        predictor: HemodynamicPredictor,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.predictor = predictor
        self.graph = build_hemodynamic_graph(predictor)


class Handler(BaseHTTPRequestHandler):
    server: HemodynamicServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path == "/api/metadata":
            self._send_json(self.server.predictor.metadata())
            return
        if path.startswith("/static/"):
            rel = unquote(path.removeprefix("/static/"))
            target = (WEB_ROOT / rel).resolve()
            if WEB_ROOT.resolve() not in target.parents:
                self._send_error(403, "Forbidden")
                return
            self._send_file(target)
            return
        self._send_error(404, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            self._send_error(404, "Not found")
            return

        try:
            payload = self._read_json()
            state = self.server.graph.invoke(
                {
                    "inputs": payload.get("inputs", {}),
                    "outputs": payload.get("outputs", []),
                    "tolerance": payload.get("tolerance", 1.0),
                    "top_k": payload.get("top_k", 3),
                }
            )
            self._send_json(state["result"])
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self._send_error(404, "Not found")
            return
        body = path.read_bytes()
        guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type or guessed)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--summary-info", default=str(DEFAULT_SUMMARY_INFO_PATH))
    args = parser.parse_args()

    predictor = HemodynamicPredictor(args.data, args.summary_info)
    server = HemodynamicServer((args.host, args.port), Handler, predictor)
    print(f"Loaded {len(predictor.rows)} cases from {predictor.data_path}")
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
