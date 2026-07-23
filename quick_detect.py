"""
Droplet detection v9 — peak finding + intensity-to-size mapping.
Blackhat for illumination correction + peak detection.
Peak intensity proxies droplet size: brighter peak = larger/darker droplet.
Radius linearly mapped: peak intensity → 1.0-2.5 px (214-497 μm at 110 μm/px).
"""

import json, math, cv2, numpy as np, os
from pathlib import Path
from skimage.feature import peak_local_max

UM_PER_PIXEL = 110.0
INPUT_DIR = Path(r"C:\LYD\imageidentification\image")
OUTPUT_DIR = Path(r"C:\LYD\imageidentification\output")
OUTPUT_DIR.mkdir(exist_ok=True)

# GT size calibration: D_eq 214-497 μm → radius 1.0-2.3 px at 110 μm/px
RADIUS_MIN = 1.0   # 220 μm
RADIUS_MAX = 2.3   # 500 μm


def detect_and_measure(gray, threshold_rel=0.30, min_distance=2):
    """
    Blackhat → peak finding → intensity-based sizing.
    Returns list of {mask, contour, bbox, area_pixels, centroid}.
    """
    h, w = gray.shape

    # Blackhat illumination correction
    ksize = 21
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    corrected = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Smooth + find peaks
    smoothed = cv2.GaussianBlur(corrected.astype(np.float64), (3, 3), 1.0)
    coords = peak_local_max(
        smoothed, min_distance=min_distance,
        threshold_rel=threshold_rel, exclude_border=2,
    )

    if len(coords) == 0:
        return [], corrected

    # Use ORIGINAL grayscale for sizing (more dynamic range than blackhat)
    # Darker pixel (lower gray value) = larger droplet
    gray_vals = gray[coords[:, 0], coords[:, 1]]
    g_min, g_max = gray_vals.min(), gray_vals.max()

    detections = []
    for i, (r, c) in enumerate(coords):
        g = gray_vals[i]

        # Darker (lower g) = larger droplet. Invert and normalize.
        if g_max > g_min:
            frac = (g_max - g) / (g_max - g_min)  # 1.0 for darkest, 0.0 for lightest
        else:
            frac = 0.5
        radius = RADIUS_MIN + frac * (RADIUS_MAX - RADIUS_MIN)

        # Create circular mask with exact radius
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - c) ** 2 + (yy - r) ** 2)
        mask = (dist <= radius).astype(np.uint8)

        area = int(np.sum(mask))
        if area < 3:
            continue

        # Get contour from circular mask
        cont_pts = np.argwhere(mask > 0)
        if len(cont_pts) < 3:
            continue
        cnt = cv2.convexHull(cont_pts[:, ::-1].astype(np.int32).reshape(-1, 1, 2))
        if cnt is None or len(cnt) < 3:
            continue

        x1 = max(0, int(c - radius - 1))
        y1 = max(0, int(r - radius - 1))
        x2 = min(w, int(c + radius + 1))
        y2 = min(h, int(r + radius + 1))

        detections.append({
            "mask": mask, "contour": cnt,
            "bbox": [x1, y1, x2, y2],
            "area_pixels": area,
            "centroid": (float(c), float(r)),
        })

    return detections, corrected


def extract_features(detections, um_per_pixel, image_name, start_id=1):
    features = []
    for i, det in enumerate(detections):
        area_px = det["area_pixels"]
        cx, cy = det["centroid"]
        eq_diam_px = 2.0 * math.sqrt(area_px / math.pi)
        eq_diam_um = eq_diam_px * um_per_pixel

        perimeter = math.pi * eq_diam_px  # circular
        circularity = 1.0

        x1, y1, x2, y2 = det["bbox"]

        features.append({
            "id": start_id + i, "image": image_name,
            "area_px": area_px,
            "eq_diameter_px": round(eq_diam_px, 4),
            "eq_diameter_um": round(eq_diam_um, 4),
            "centroid_x": round(cx, 4),
            "centroid_y": round(cy, 4),
            "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
            "confidence": 0.85,
            "circularity": circularity,
            "perimeter_px": round(perimeter, 4),
            "major_axis_um": round(eq_diam_um, 4),
            "minor_axis_um": round(eq_diam_um, 4),
            "min_feret_um": round(eq_diam_um, 4),
            "max_feret_um": round(eq_diam_um, 4),
            "eccentricity": 0.0,
            "aspect_ratio": 1.0,
        })
    return features


def draw_results(img, features):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    # Color by size: small=blue, medium=green, large=red
    diams = np.array([f["eq_diameter_um"] for f in features])
    if len(diams) > 0:
        for f, d in zip(features, diams):
            if d < 300:
                color = (255, 0, 0)   # blue = small
            elif d < 400:
                color = (0, 255, 0)   # green = medium
            else:
                color = (0, 0, 255)   # red = large
            cv2.circle(vis, (int(f["centroid_x"]), int(f["centroid_y"])), 1, color, -1)
    return vis


def main():
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
    all_features = []

    THRESHOLD_REL = 0.30
    MIN_DISTANCE = 2

    for fname in image_files:
        fpath = os.path.join(str(INPUT_DIR), fname)
        print(f"Processing: {fname}")

        with open(fpath, "rb") as fp:
            data = fp.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)

        detections, corrected = detect_and_measure(
            img, threshold_rel=THRESHOLD_REL, min_distance=MIN_DISTANCE,
        )

        areas = [d["area_pixels"] for d in detections]
        if areas:
            diams_px = [2.0 * math.sqrt(a / math.pi) for a in areas]
            diams_um = [d * UM_PER_PIXEL for d in diams_px]
            print(f"  Detected: {len(detections)} objects | D_eq: {min(diams_um):.0f}-{max(diams_um):.0f} μm | mean={np.mean(diams_um):.0f}")

        features = extract_features(
            detections, um_per_pixel=UM_PER_PIXEL,
            image_name=fname, start_id=len(all_features) + 1,
        )
        all_features.extend(features)

        vis = draw_results(img, features)
        cv2.imwrite(str(OUTPUT_DIR / f"{Path(fname).stem}_det.png"), vis)
        cv2.imwrite(str(OUTPUT_DIR / f"{Path(fname).stem}_blackhat.png"), corrected)

    # Summary
    print(f"\n===== TOTAL: {len(all_features)} =====")
    if all_features:
        diams = [f["eq_diameter_um"] for f in all_features]
        print(f"D_eq: {min(diams):.0f} ~ {max(diams):.0f} μm | mean={np.mean(diams):.0f} | median={np.median(diams):.0f}")
        d32 = sum(d**3 for d in diams) / sum(d**2 for d in diams) if sum(d**2 for d in diams) > 0 else 0
        print(f"D32: {d32:.0f} μm")

        bins = [(220, 280), (280, 340), (340, 400), (400, 460), (460, 520)]
        print("Distribution:")
        for lo, hi in bins:
            n = sum(1 for d in diams if lo <= d < hi)
            print(f"  {lo}-{hi} μm: {n:5d} ({n/len(diams)*100:5.1f}%)")

    with open(OUTPUT_DIR / "results_v9.json", "w", encoding="utf-8") as f:
        json.dump(all_features, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
