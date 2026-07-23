"""
API server for the PIV Image Cropper.
Serves the cropper HTML + API endpoints for folder listing, image loading, and crop saving.
"""

import http.server
import json
import os
import urllib.parse
import cv2
import numpy as np
from io import BytesIO
from pathlib import Path

IMAGE_ROOT = Path(r"C:\LYD\imageidentification\image")
OUTPUT_DIR = Path(r"C:\LYD\imageidentification\output")
CROPS_FILE = OUTPUT_DIR / "crops.json"

# Load saved crops
if CROPS_FILE.exists():
    with open(CROPS_FILE) as f:
        saved_crops = json.load(f)
else:
    saved_crops = {}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(OUTPUT_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/api/folders':
            self.send_json_response(self._get_folders())

        elif parsed.path == '/api/images':
            params = urllib.parse.parse_qs(parsed.query)
            folder = params.get('folder', [''])[0]
            self.send_json_response(self._get_images(folder))

        elif parsed.path == '/api/image':
            params = urllib.parse.parse_qs(parsed.query)
            folder = params.get('folder', [''])[0]
            fname = params.get('file', [''])[0]
            self._serve_image(folder, fname)

        elif parsed.path == '/api/crops':
            self.send_json_response({"crops": saved_crops})

        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/api/save_crop':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            folder = data['folder'].replace('/', os.sep)
            saved_crops[folder] = {
                'x1': data['x1'], 'y1': data['y1'],
                'x2': data['x2'], 'y2': data['y2'],
            }
            with open(CROPS_FILE, 'w') as f:
                json.dump(saved_crops, f, indent=2)
            self.send_json_response({"status": "ok"})

        elif parsed.path == '/api/delete_crop':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            folder = data['folder']
            saved_crops.pop(folder, None)
            with open(CROPS_FILE, 'w') as f:
                json.dump(saved_crops, f, indent=2)
            self.send_json_response({"status": "ok"})

        else:
            self.send_error(404)

    def _get_folders(self):
        folders = []
        for parent in ['H0', 'H1']:
            pdir = IMAGE_ROOT / parent
            if pdir.exists():
                for sub in sorted(pdir.iterdir()):
                    if sub.is_dir():
                        # Use forward slashes for URL compatibility
                        rel = str(sub.relative_to(IMAGE_ROOT)).replace('\\', '/')
                        folders.append(rel)
        return {"folders": folders}

    def _get_images(self, folder):
        # Convert back from forward-slash to OS path
        folder = folder.replace('/', os.sep)
        fdir = IMAGE_ROOT / folder
        images = []
        if fdir.exists():
            for f in sorted(fdir.iterdir()):
                if f.suffix.lower() == '.bmp':
                    images.append(f.name)
        return {"images": images}

    def _serve_image(self, folder, fname):
        folder = folder.replace('/', os.sep)
        fpath = IMAGE_ROOT / folder / fname
        if not fpath.exists():
            self.send_error(404)
            return

        img = cv2.imread(str(fpath))
        if img is None:
            self.send_error(500)
            return

        # Convert to JPEG for faster transfer
        _, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(jpg))
        self.send_header('Cache-Control', 'max-age=3600')
        self.end_headers()
        self.wfile.write(jpg.tobytes())

    def send_json_response(self, data):
        body = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress logs


if __name__ == "__main__":
    port = 8080
    server = http.server.HTTPServer(('', port), Handler)
    print(f"Cropper server running at http://localhost:{port}/cropper.html")
    server.serve_forever()
