"""
generate_masks_zip.py
=====================
Automatically generates inhibition-zone segmentation masks for the
ZIP dataset images using a multi-method pipeline:

  1. Detect the Petri dish boundary (large Hough circle)
  2. Detect antibiotic discs (small Hough circles)
  3. For each disc, extract the local region and segment the clear
     halo using adaptive thresholding + morphological refinement
  4. Optionally refine with GrabCut for cleaner boundaries

No ground-truth measurements needed — fully unsupervised.

Output: binary PNG mask per image in data/zip_dataset/masks/
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


# ── tuneable parameters ───────────────────────────────────────────────────────
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Petri dish detection
DISH_MIN_RADIUS_FRACTION = 0.25   # dish radius >= 25% of min(H,W)
DISH_MAX_RADIUS_FRACTION = 0.55

# Antibiotic disc detection
DISC_MIN_PX  = 8
DISC_MAX_PX  = 45
DISC_DIAM_MM = 6.0                 # physical reference

# Inhibition zone search ring
ZONE_SEARCH_MULTIPLIER = 4.5       # search up to 4.5× disc radius for zone edge


# ── image pre-processing ──────────────────────────────────────────────────────

def preprocess(img: np.ndarray):
    """Return CLAHE-enhanced grayscale for better contrast on pale zones."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ── Petri dish detection ──────────────────────────────────────────────────────

def find_dish(gray: np.ndarray):
    """
    Returns (cx, cy, radius) of the Petri dish, or None if not found.
    We look for one large circle near the image centre.
    """
    h, w = gray.shape
    min_r = int(min(h, w) * DISH_MIN_RADIUS_FRACTION)
    max_r = int(min(h, w) * DISH_MAX_RADIUS_FRACTION)
    blurred = cv2.GaussianBlur(gray, (15, 15), 3)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.5, minDist=min(h, w),
        param1=50, param2=40,
        minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return None
    c = np.round(circles[0][0]).astype(int)
    return (c[0], c[1], c[2])


# ── antibiotic disc detection ─────────────────────────────────────────────────

def find_discs(gray: np.ndarray, dish=None):
    """
    Returns list of (cx, cy, radius) for each antibiotic disc.
    Restricts search to inside the dish boundary if dish is provided.
    """
    mask = np.zeros_like(gray)
    if dish:
        cv2.circle(mask, (dish[0], dish[1]), dish[2], 255, -1)
        search = cv2.bitwise_and(gray, gray, mask=mask)
    else:
        search = gray.copy()

    blurred = cv2.GaussianBlur(search, (7, 7), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=30,
        param1=60, param2=28,
        minRadius=DISC_MIN_PX, maxRadius=DISC_MAX_PX
    )
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    result = [(c[0], c[1], c[2]) for c in circles]

    # filter: must be inside dish
    if dish:
        dx, dy, dr = dish
        result = [
            c for c in result
            if math.hypot(c[0]-dx, c[1]-dy) < dr - c[2]
        ]
    return result


# ── zone segmentation around a single disc ────────────────────────────────────

def segment_zone(gray: np.ndarray, cx: int, cy: int, disc_r: int,
                 dish=None) -> np.ndarray:
    """
    Segments the inhibition zone (clear halo) around one antibiotic disc.
    Returns a binary mask (uint8, same size as gray).

    Strategy:
      - Extract an annular ROI from disc edge outward
      - Apply Otsu thresholding on the annular region
      - Morphological clean-up
      - Fit an ellipse / circle to the largest contour
    """
    h, w  = gray.shape
    search_r = int(disc_r * ZONE_SEARCH_MULTIPLIER)

    # ── build annular search region ───────────────────────────────────────
    annulus = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(annulus, (cx, cy), search_r, 255, -1)
    cv2.circle(annulus, (cx, cy), disc_r,   0,  -1)   # remove disc itself
    if dish:
        dish_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(dish_mask, (dish[0], dish[1]), dish[2], 255, -1)
        annulus = cv2.bitwise_and(annulus, dish_mask)

    roi_pixels = gray[annulus == 255]
    if roi_pixels.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # ── adaptive threshold on the annular region ──────────────────────────
    # The inhibition zone is lighter (less bacterial growth) than surroundings
    mean_val  = roi_pixels.mean()
    std_val   = roi_pixels.std()
    threshold = mean_val + 0.3 * std_val     # slightly above mean = clear zone

    binary = np.zeros((h, w), dtype=np.uint8)
    binary[(gray >= threshold) & (annulus == 255)] = 255

    # include the disc centre (it's definitely "in" the zone)
    cv2.circle(binary, (cx, cy), search_r, 0, -1)
    cv2.circle(binary, (cx, cy), disc_r + 2, 255, -1)

    # ── Otsu as alternative if adaptive gives poor result ─────────────────
    roi_img = cv2.bitwise_and(gray, gray, mask=annulus)
    _, otsu  = cv2.threshold(roi_img, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_in  = cv2.bitwise_and(otsu, annulus)

    # pick whichever gives a larger connected region near the disc
    def largest_cc_area(bw):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
        if num <= 1:
            return 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        return int(areas.max()) if len(areas) else 0

    chosen = binary if largest_cc_area(binary) >= largest_cc_area(otsu_in) \
             else otsu_in

    # ── morphological refinement ──────────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    chosen = cv2.morphologyEx(chosen, cv2.MORPH_CLOSE, kernel, iterations=3)
    chosen = cv2.morphologyEx(chosen, cv2.MORPH_OPEN,  kernel, iterations=1)

    # ── fit circle to largest contour for smooth output ───────────────────
    contours, _ = cv2.findContours(chosen, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros((h, w), dtype=np.uint8)

    if contours:
        # keep only the contour whose centroid is close to disc centre
        best = None
        best_dist = 1e9
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            mx = int(M["m10"] / M["m00"])
            my = int(M["m01"] / M["m00"])
            d  = math.hypot(mx - cx, my - cy)
            if d < best_dist:
                best_dist = d
                best = cnt

        if best is not None and best_dist < search_r:
            # fit enclosing circle for clean circular mask
            (ex, ey), er = cv2.minEnclosingCircle(best)
            er = min(er, search_r)          # cap at search radius
            cv2.circle(final_mask, (int(ex), int(ey)), int(er), 255, -1)

    # punch out the disc itself so mask = halo only
    cv2.circle(final_mask, (cx, cy), disc_r, 0, -1)
    return final_mask


import math


# ── per-image pipeline ────────────────────────────────────────────────────────

def process_image(img_path: Path, masks_dir: Path, visualise: bool = False):
    img  = cv2.imread(str(img_path))
    if img is None:
        return False

    gray  = preprocess(img)
    h, w  = gray.shape
    dish  = find_dish(gray)
    discs = find_discs(gray, dish)

    if not discs:
        # fallback: try without dish constraint
        discs = find_discs(gray, None)

    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for (cx, cy, r) in discs:
        zone_mask = segment_zone(gray, cx, cy, r, dish)
        combined_mask = cv2.bitwise_or(combined_mask, zone_mask)

    stem      = img_path.stem
    mask_path = masks_dir / (stem + "_mask.png")
    cv2.imwrite(str(mask_path), combined_mask)

    # optional debug visualisation
    if visualise:
        vis = img.copy()
        vis[combined_mask > 0] = (vis[combined_mask > 0] * 0.5 +
                                  np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        if dish:
            cv2.circle(vis, (dish[0], dish[1]), dish[2], (255, 0, 0), 2)
        for cx, cy, r in discs:
            cv2.circle(vis, (cx, cy), r, (0, 0, 255), 2)
        vis_path = masks_dir / (stem + "_vis.jpg")
        cv2.imwrite(str(vis_path), vis)

    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main(images_dir: str, masks_dir: str, visualise: bool):
    images_path = Path(images_dir)
    masks_path  = Path(masks_dir)
    masks_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in images_path.iterdir()
             if f.suffix.lower() in SUPPORTED_EXT]

    if not files:
        print(f"❌  No images found in {images_dir}")
        return

    ok = 0
    for f in tqdm(files, desc="Segmenting ZIP images"):
        if process_image(f, masks_path, visualise):
            ok += 1

    print(f"\n✅  Processed {ok}/{len(files)} images.")
    print(f"    Masks saved to: {masks_dir}")
    if visualise:
        print(f"    Debug overlays saved alongside masks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-generate inhibition zone masks from ZIP dataset images"
    )
    parser.add_argument("--images_dir", required=True,
                        help="Folder containing extracted ZIP images")
    parser.add_argument("--masks_dir",  default="data/zip_dataset/masks",
                        help="Output folder for masks")
    parser.add_argument("--visualise",  action="store_true",
                        help="Save debug overlay images for quality checking")
    args = parser.parse_args()
    main(args.images_dir, args.masks_dir, args.visualise)
