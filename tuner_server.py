"""Parameter tuner server. Run: python tuner_server.py"""
import http.server, json, urllib.parse, cv2, numpy as np
from pathlib import Path

# Load pre-built median background from cached .npy file for instant startup
MEDIAN_BG_NPY = Path(__file__).parent / 'output' / 'tuner_median.npy'
MEDIAN_BG = np.load(str(MEDIAN_BG_NPY)) if MEDIAN_BG_NPY.exists() else None
print(f'Median bg loaded from cache: {MEDIAN_BG.shape if MEDIAN_BG is not None else "None"}')

# Full tunable median detection — all parameters adjustable from UI
def detect_median(gray, median_bg, T=5, MD=8, MA=6, MO=1, PT=0.03, SF=1.0):
    import math
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    h,w = gray.shape
    # Auto-detect droplet polarity
    res_a = cv2.subtract(median_bg, gray)
    res_b = cv2.subtract(gray, median_bg)
    residual = res_a if (res_a > T).sum() > (res_b > T).sum() else res_b
    binary = residual > T
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if MO > 0:
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, k, iterations=MO)
    binary = binary.astype(bool)
    dist_map = ndi.distance_transform_edt(binary)
    coords = peak_local_max(dist_map, min_distance=MD, threshold_rel=PT, exclude_border=5)
    if len(coords) == 0:
        return []
    markers = np.zeros((h,w), dtype=np.int32)
    for i,(r,c) in enumerate(coords): markers[r,c]=i+1
    labels = watershed(-dist_map, markers, mask=binary)
    dets = []
    for lid in range(1, labels.max()+1):
        region = labels == lid
        n_px = int(region.sum())
        if n_px < MA or n_px > 350: continue
        ys,xs = np.where(region)
        if len(ys)<5: continue
        if ys.min()<=5 or xs.min()<=5 or ys.max()>=h-6 or xs.max()>=w-6: continue
        d_eq_px = round(2*math.sqrt(n_px/math.pi),1)
        vis_r = max(1,int(d_eq_px * SF / 2))
        cx,cy = xs.mean(),ys.mean()
        pts = cv2.ellipse2Poly((int(cx),int(cy)), (vis_r,vis_r), 0, 0, 360, 32)
        dets.append({'id':len(dets),'contour':pts.tolist(),
                     'center':[round(float(cx),1),round(float(cy),1)],
                     'axes':[d_eq_px,d_eq_px],'angle':0.0,'area_px':n_px,'type':'ellipse'})
    dets.sort(key=lambda d:d['area_px'],reverse=True)
    return dets


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory='output', **kw)

    def do_POST(self):
        if self.path == '/api/tuner_detect':
            cl = int(self.headers.get('Content-Length', 0))
            params = json.loads(self.rfile.read(cl))
            use_median = params.pop('use_median', True)
            with open(r'C:\LYD\imageidentification\output\cropped\H0\50cst-H0110rpm0.00175kw\0000.bmp','rb') as f:
                gray = cv2.imdecode(np.frombuffer(f.read(),np.uint8), cv2.IMREAD_GRAYSCALE)
            if use_median and MEDIAN_BG is not None:
                # Extract custom median params from request
                T = params.pop('T', 6)
                MD = params.pop('MD', 9)
                MA = params.pop('MA', 6)
                MO = params.pop('MO', 1)
                PT = params.pop('PT', 0.04)
                SF = params.pop('SF', 1.0)
                dets = detect_median(gray, MEDIAN_BG, T=T, MD=MD, MA=MA, MO=MO, PT=PT, SF=SF)
            else:
                from app.main_app import detect_droplets
                dets = detect_droplets(gray, **params)
            body = json.dumps({'droplets': dets}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a): pass

if __name__ == '__main__':
    print('Tuner: http://localhost:8080/tuner.html')
    http.server.HTTPServer(('', 8080), H).serve_forever()
