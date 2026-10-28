"""Dev server for the dashboard. Identical to http.server except that it tells
the browser not to cache, so an edit to app.js is visible on reload instead of
hours later."""

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "200" not in fmt % args:
            super().log_message(fmt, *args)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    root = Path(__file__).parent / "docs"
    print(f"serving {root} on http://localhost:{port}")
    ThreadingHTTPServer(
        ("127.0.0.1", port), partial(NoCache, directory=str(root))
    ).serve_forever()
