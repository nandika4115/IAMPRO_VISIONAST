"""
predict.py
==========
Run VisionAST on a new plate image:
  1. Segment inhibition zones with trained U-Net
  2. Detect antibiotic discs (Hough) as physical references
  3. Convert pixel measurements → mm using the 6 mm disc as ruler
  4. Look up CLSI / EUCAST breakpoints → S / I / R

Usage:
    python src/predict.py \
        --model_path  models/best_model.pth \
        --image       path/to/plate.jpg \
        --output_dir  outputs/ \
        --model_size  small \
        --guideline   EUCAST
"""

import argparse
import json
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model


# ── CLSI / EUCAST breakpoints (zone diameter in mm) ──────────────────────────
# Format:  antibiotic_disc_label -> {S_min, R_max}
# Zones >= S_min  → Susceptible
# Zones <= R_max  → Resistant
# Between         → Intermediate
# Source: EUCAST 2024 / CLSI M02-A13

BREAKPOINTS = {
    "EUCAST": {
        "AMP":  {"S": 14, "R": 13},   # Ampicillin
        "AMX":  {"S": 19, "R": 18},   # Amoxicillin
        "CIP":  {"S": 25, "R": 24},   # Ciprofloxacin
        "GEN":  {"S": 15, "R": 14},   # Gentamicin
        "TET":  {"S": 18, "R": 17},   # Tetracycline
        "SXT":  {"S": 16, "R": 13},   # Trimethoprim-sulfamethoxazole
        "CTX":  {"S": 20, "R": 17},   # Cefotaxime
        "MEM":  {"S": 22, "R": 21},   # Meropenem
        "IPM":  {"S": 22, "R": 21},   # Imipenem
        "VAN":  {"S": 17, "R": 14},   # Vancomycin
        "ERY":  {"S": 18, "R": 17},   # Erythromycin
        "CLI":  {"S": 22, "R": 21},   # Clindamycin
        "DEFAULT": {"S": 16, "R": 13},
    },
    "CLSI": {
        "AMP":  {"S": 17, "R": 13},
        "AMX":  {"S": 17, "R": 13},
        "CIP":  {"S": 21, "R": 15},
        "GEN":  {"S": 15, "R": 12},
        "TET":  {"S": 19, "R": 14},
        "SXT":  {"S": 16, "R": 10},
        "CTX":  {"S": 26, "R": 22},
        "MEM":  {"S": 23, "R": 19},
        "IPM":  {"S": 23, "R": 19},
        "VAN":  {"S": 17, "R": 14},
        "ERY":  {"S": 23, "R": 13},
        "CLI":  {"S": 21, "R": 14},
        "DEFAULT": {"S": 17, "R": 13},
    },
}

DISC_DIAMETER_MM = 6.0


# ── helpers ───────────────────────────────────────────────────────────────────

def classify(zone_mm: float, disc_label: str, guideline: str) -> str:
    bp_table = BREAKPOINTS.get(guideline, BREAKPOINTS["EUCAST"])
    bp = bp_table.get(disc_label.upper(),
                      bp_table.get("DEFAULT", {"S": 16, "R": 13}))
    if zone_mm >= bp["S"]:
        return "S"
    if zone_mm <= bp["R"]:
        return "R"
    return "I"


def find_discs(gray: np.ndarray):
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=30,
        param1=60, param2=28,
        minRadius=8, maxRadius=45,
    )
    if circles is None:
        return []
    return [tuple(np.round(c).astype(int))
            for c in circles[0]]   # (cx, cy, r)


def pixels_to_mm(zone_radius_px: float, disc_radius_px: float) -> float:
    px_per_mm = (disc_radius_px * 2) / DISC_DIAMETER_MM
    return (zone_radius_px * 2) / px_per_mm   # return diameter


# ── segmentation ─────────────────────────────────────────────────────────────

def segment(model, img_rgb: np.ndarray, img_size: int, device: str
            ) -> np.ndarray:
    """Returns binary mask (H×W, uint8) at original image resolution."""
    h, w   = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, (img_size, img_size))
    tensor  = torch.from_numpy(resized.transpose(2,0,1)).float() / 255.0
    tensor  = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[:, 1]   # foreground probability
        pred   = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255

    mask = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


# ── zone measurement from mask + disc ────────────────────────────────────────

def measure_zones(mask: np.ndarray, discs, img_rgb: np.ndarray,
                  guideline: str):
    """
    For each disc, find the connected component in mask closest to it,
    fit a circle, convert to mm and classify.
    Returns list of result dicts.
    """
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    results = []
    vis     = img_rgb.copy()

    # draw mask overlay
    green_layer = np.zeros_like(img_rgb)
    green_layer[mask > 127] = [0, 200, 80]
    vis = cv2.addWeighted(vis, 0.7, green_layer, 0.3, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for i, (cx, cy, disc_r) in enumerate(discs):
        # median disc radius as reference (px per mm)
        median_disc_r = float(np.median([d[2] for d in discs]))

        # find nearest contour centroid
        best_cnt  = None
        best_dist = 1e9
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            mx = M["m10"] / M["m00"]
            my = M["m01"] / M["m00"]
            d  = math.hypot(mx - cx, my - cy)
            if d < best_dist:
                best_dist = d
                best_cnt  = cnt

        if best_cnt is None or best_dist > disc_r * 5:
            # no zone found near this disc
            results.append({
                "disc_index": i+1,
                "disc_cx": int(cx), "disc_cy": int(cy),
                "zone_diameter_mm": 0.0,
                "classification": "R",
                "note": "no zone detected",
            })
            continue

        (ex, ey), er = cv2.minEnclosingCircle(best_cnt)
        zone_mm  = pixels_to_mm(er, median_disc_r)
        sir      = classify(zone_mm, "DEFAULT", guideline)

        colour = {"S": (0,200,0), "I": (255,165,0), "R": (220,0,0)}[sir]
        cv2.circle(vis, (int(ex), int(ey)), int(er), colour, 2)
        cv2.circle(vis, (cx, cy), disc_r, (255,255,255), 2)
        cv2.putText(vis, f"{zone_mm:.1f}mm {sir}",
                    (int(ex)-30, int(ey)-int(er)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        results.append({
            "disc_index":        i+1,
            "disc_cx":           int(cx),
            "disc_cy":           int(cy),
            "zone_radius_px":    float(er),
            "zone_diameter_mm":  round(float(zone_mm), 2),
            "classification":    sir,
        })

    return results, vis


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = build_model(size=args.model_size, device=device)
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"❌  Could not read image: {args.image}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # segmentation
    mask  = segment(model, img_rgb, args.img_size, device)

    # disc detection
    discs = find_discs(gray)
    if not discs:
        print("⚠️  No antibiotic discs detected via Hough. "
              "Results will use image-centre fallback.")
        h, w = gray.shape
        discs = [(w//2, h//2, 20)]

    # measurement & classification
    results, vis = measure_zones(mask, discs, img_rgb, args.guideline)

    # save outputs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem      = Path(args.image).stem
    vis_bgr   = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    mask_path = out / f"{stem}_mask.png"
    vis_path  = out / f"{stem}_result.jpg"
    json_path = out / f"{stem}_measurements.json"

    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(vis_path),  vis_bgr)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n── Results ──")
    for r in results:
        print(f"  Disc {r['disc_index']}: "
              f"{r['zone_diameter_mm']:.1f} mm  → {r['classification']}")
    print(f"\nSaved: {vis_path}, {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionAST inference")
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--image",       required=True)
    parser.add_argument("--output_dir",  default="outputs/")
    parser.add_argument("--model_size",  choices=["full","small"],
                        default="small")
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--guideline",   choices=["EUCAST","CLSI"],
                        default="EUCAST")
    args = parser.parse_args()
    main(args)
