"""
Review server for PIV droplet detection.
Serves static files + detections API on port 8080.
"""

import http.server
import json
import os
import urllib.parse
import cv2
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
DETECTIONS_DIR = OUTPUT_DIR / "detections"
CROPPED_DIR = Path(r"C:\LYD\imageidentification\output\cropped")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(OUTPUT_DIR), **kwargs)

    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        if p.path == '/api/images':
            q = urllib.parse.parse_qs(p.query)
            folder = q.get('folder', [''])[0]
            self._json_images(folder)
        elif p.path == '/api/detections':
            q = urllib.parse.parse_qs(p.query)
            folder = q.get('folder', [''])[0]
            self._json_detections(folder)
        elif p.path == '/api/image_raw':
            q = urllib.parse.parse_qs(p.query)
            folder = q.get('folder', [''])[0]
            fname = q.get('file', [''])[0]
            self._serve_image(folder, fname)
        elif p.path == '/api/image_overlay':
            q = urllib.parse.parse_qs(p.query)
            folder = q.get('folder', [''])[0]
            fname = q.get('file', [''])[0]
            self._serve_overlay(folder, fname)
        else:
            super().do_GET()

    def do_POST(self):
        p = urllib.parse.urlparse(self.path)
        cl = int(self.headers.get('Content-Length', 0))
        data = json.loads(self.rfile.read(cl)) if cl > 0 else {}

        if p.path == '/api/save':
            self._save(data)
        else:
            self.send_error(404)

    def _json_images(self, folder):
        fdir = CROPPED_DIR / folder.replace('/', os.sep)
        imgs = sorted([f for f in os.listdir(str(fdir)) if f.lower().endswith('.bmp')]) if fdir.exists() else []
        self._send_json({"images": imgs})

    def _json_detections(self, folder):
        det_name = folder.replace('/', '__').replace(os.sep, '__')
        det_file = DETECTIONS_DIR / f"{det_name}.json"
        if det_file.exists():
            with open(det_file) as f:
                data = json.load(f)
        else:
            data = {"images": {}}
        self._send_json(data)

    def _serve_image(self, folder, fname):
        fpath = CROPPED_DIR / folder.replace('/', os.sep) / fname
        if not fpath.exists():
            self.send_error(404); return
        img = cv2.imread(str(fpath))
        if img is None:
            self.send_error(500); return
        _, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(jpg))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(jpg.tobytes())

    def _serve_overlay(self, folder, fname):
        fpath = CROPPED_DIR / folder.replace('/', os.sep) / fname
        if not fpath.exists():
            self.send_error(404); return
        img = cv2.imread(str(fpath))
        if img is None:
            self.send_error(500); return

        det_name = folder.replace('/', '__').replace(os.sep, '__')
        det_file = DETECTIONS_DIR / f"{det_name}.json"
        if det_file.exists():
            with open(det_file) as f:
                det_data = json.load(f)
            img_data = det_data.get("images", {}).get(fname, {})
            colors = [(0,255,0),(255,255,0),(0,255,255),(255,0,255),
                      (0,128,255),(255,128,0),(128,255,0),(0,0,255)]
            for d in img_data.get("droplets", []):
                pts = np.array(d["contour"], dtype=np.int32).reshape(-1,1,2)
                cv2.drawContours(img, [pts], -1, colors[d["id"] % 8], 2)

        _, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(jpg))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(jpg.tobytes())

    def _save(self, data):
        folder = data.get('folder', '')
        fname = data.get('file', '')
        droplets = data.get('droplets', [])
        det_name = folder.replace('/', '__').replace(os.sep, '__')
        det_file = DETECTIONS_DIR / f"{det_name}.json"
        if det_file.exists():
            with open(det_file) as f:
                det_data = json.load(f)
        else:
            det_data = {"images": {}}
        det_data["images"][fname] = {"droplets": droplets, "count": len(droplets)}
        det_data["total_droplets"] = sum(v.get("count",0) for v in det_data["images"].values())
        with open(det_file, 'w') as f:
            json.dump(det_data, f, indent=2)
        self._send_json({"status": "ok"})

    def _send_json(self, data):
        b = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(b))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    print(f"Review server: http://localhost:8080/review.html")
    http.server.HTTPServer(('', 8080), Handler).serve_forever()
