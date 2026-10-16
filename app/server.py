"""Web app for the Heart Disease Prediction System.

Serves the static UI in app/static/ and one JSON endpoint:
    POST /api/predict   body: the 13 patient fields -> probability, risk level,
                        recommendations

Run from the project root:
    python app/server.py            # http://localhost:8501
    python app/server.py --port 9000
"""
import argparse
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.hdp_system.predict import load_model, predict  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            self.send_json(200, predict(body))
        except ValueError as e:  # includes bad JSON and failed validation
            self.send_json(400, {"error": str(e)})
        except Exception as e:
            print(f"prediction failed: {e!r}")
            self.send_json(500, {"error": "Prediction failed. Check the server log."})

    def send_json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if self.command == "POST" or (args and str(args[1]).startswith(("4", "5"))):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    load_model()  # fail fast if the model file is missing
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Heart Disease Prediction System: http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
