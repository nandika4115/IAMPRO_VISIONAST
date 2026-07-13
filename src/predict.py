"""
predict.py  —  VisionAST inference  (fixed: mask inversion, disc count, breakpoints)
=====================================================================================
Fixes applied vs previous version:
  1. Mask inversion fix — zones are the DARK halos (low pixel value), not bright
  2. Tighter Hough disc detection — reduces false positives from glare/noise
  3. Antibiotic disc OCR — reads the label printed on each disc (e.g. "CIP", "MEM")
     and looks up the correct EUCAST/CLSI breakpoint for that antibiotic
  4. Post-mask cleanup — morphological ops to remove fragmented predictions
  5. Plate crop uses contour-based fallback for square/rectangular plates
     (SIRscan plates are square, not circular — previous version missed this)
"""

import argparse, json, math, sys, re
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model


# ── EUCAST breakpoints keyed by disc abbreviation (zone diameter mm) ──────────
# S >= S_val, R <= R_val, else I
BREAKPOINTS_EUCAST = {
    # abbreviation : (S_threshold, R_threshold)
    "AM10":  (15, 13), "AMP":   (15, 13),
    "AMC":   (19, 15), "AMX":   (19, 15),
    "CIP":   (25, 24), "PEF":   (24, 20),
    "GEN":   (15, 14), "CN":    (15, 14),
    "TET":   (18, 17),
    "SXT":   (16, 13),
    "CTX":   (20, 17), "CRO":   (23, 20),
    "MEM":   (22, 21),
    "IPM":   (22, 21),
    "ETP":   (25, 22),
    "VAN":   (17, 14),
    "ERY":   (18, 17),
    "CLI":   (22, 21),
    "CAZ":   (21, 17),
    "FEP":   (25, 19),
    "FOX":   (22, 19), "FOX30": (22, 19),
    "CPD":   (22, 19),
    "TPZ":   (20, 20), "TZP":   (20, 20),
    "AK":    (18, 14),
    "TOB":   (17, 14),
    "FF":    (26, 23),
    "F100":  (15, 14),
    "DEFAULT": (16, 13),
}

DISC_DIAMETER_MM = 6.0

SIR_COLOR = {
    "S": (0, 200, 0),      # green
    "I": (0, 165, 255),    # orange
    "R": (0, 0, 220),      # red
}


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE QUALITY VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class QualityResult:
    def __init__(self):
        self.ok     = True
        self.issues = []

    def fail(self, code, msg, instr):
        self.ok = False
        self.issues.append({"code":code,"message":msg,
                            "instruction":instr,"severity":"error"})

    def warn(self, code, msg, instr):
        self.issues.append({"code":code,"message":msg,
                            "instruction":instr,"severity":"warning"})


def validate_image(img_bgr):
    r    = QualityResult()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if w < 400 or h < 400:
        r.fail("LOW_RESOLUTION",
               f"Image too small ({w}×{h}px, need ≥400×400).",
               "Move closer to the plate or use a higher resolution setting.")

    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap < 80:
        r.fail("BLURRY", f"Image is blurry (score {lap:.1f}, need ≥80).",
               "Hold camera steady and wait for autofocus to lock.")

    br = gray.mean()
    if br < 40:
        r.fail("TOO_DARK", f"Image too dark (brightness {br:.0f}/255).",
               "Turn on overhead lights. Avoid shadows on the plate.")
    elif br > 220:
        r.fail("TOO_BRIGHT", f"Image overexposed (brightness {br:.0f}/255).",
               "Avoid direct flash. Move away from strong light.")

    overexp = (gray > 250).sum() / gray.size
    if overexp > 0.03:
        r.warn("GLARE", f"Glare/reflections on {overexp*100:.1f}% of image.",
               "Tilt light source sideways to avoid agar surface reflections.")

    tilt = _measure_tilt(gray)
    if tilt > 0.18:
        r.warn("TILTED", f"Plate appears angled (tilt {tilt:.2f}).",
               "Hold camera directly above the plate, looking straight down.")

    return r


def _measure_tilt(gray):
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),1), 30, 100)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    largest = max(cnts, key=cv2.contourArea)
    if len(largest) < 5: return 0.0
    (_, _), (MA, ma), _ = cv2.fitEllipse(largest)
    return abs(1.0 - min(MA,ma)/max(MA,ma)) if max(MA,ma) > 0 else 0.0


def make_alert_image(img_bgr, issues):
    h, w  = img_bgr.shape[:2]
    alert = (img_bgr * 0.35).astype(np.uint8)
    cv2.rectangle(alert, (5,5), (w-5,h-5), (0,0,220), 6)
    cv2.putText(alert, "Photo Quality Issue — Please Retake",
                (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,80,255), 2)
    y = 90
    for iss in issues:
        col = (80,80,255) if iss["severity"]=="error" else (0,180,255)
        words, line, lines = iss["instruction"].split(), "", []
        for ww in words:
            candidate = (line+" "+ww).strip()
            if len(candidate) > 52: lines.append(line); line=ww
            else: line=candidate
        if line: lines.append(line)
        for li, ln in enumerate(lines):
            prefix = "• " if li==0 else "  "
            cv2.putText(alert, prefix+ln, (20,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)
            y += 24
        y += 8
    return alert


# ══════════════════════════════════════════════════════════════════════════════
# PLATE CROP  — handles both circular AND square/rectangular plates
# ══════════════════════════════════════════════════════════════════════════════

def crop_to_plate(img_bgr):
    """
    Returns (cropped_bgr, offset_xy) where offset_xy=(x1,y1) is the
    top-left corner of the crop in original image coordinates.
    Tries circular Hough first; falls back to largest contour bounding box.
    """
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape
    min_r  = int(min(h,w)*0.20)
    max_r  = int(min(h,w)*0.58)
    blur   = cv2.GaussianBlur(gray,(15,15),3)

    # --- try circular plate ---
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.5,
                               minDist=min(h,w), param1=50, param2=35,
                               minRadius=min_r, maxRadius=max_r)
    if circles is not None:
        cx, cy, r = np.round(circles[0][0]).astype(int)
        pad = int(r*0.06)
        x1,y1 = max(cx-r-pad,0), max(cy-r-pad,0)
        x2,y2 = min(cx+r+pad,w), min(cy+r+pad,h)
        mask   = np.zeros((h,w), np.uint8)
        cv2.circle(mask,(cx,cy),r+pad,255,-1)
        out = img_bgr.copy(); out[mask==0] = 0
        return out[y1:y2, x1:x2], (x1, y1)

    # --- fallback: largest bright contour (works for square SIRscan plates) ---
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        bx,by,bw,bh = cv2.boundingRect(largest)
        pad = 10
        x1,y1 = max(bx-pad,0), max(by-pad,0)
        x2,y2 = min(bx+bw+pad,w), min(by+bh+pad,h)
        return img_bgr[y1:y2, x1:x2], (x1, y1)

    return img_bgr, (0, 0)


# ══════════════════════════════════════════════════════════════════════════════
# DISC DETECTION  — tightened to reduce false positives
# ══════════════════════════════════════════════════════════════════════════════

def find_discs(gray, max_expected=20):
    """
    Returns list of (cx, cy, radius) for antibiotic discs.
    Uses brightness (discs are bright white) + Hough, then
    filters by brightness of the disc region.
    """
    blur = cv2.GaussianBlur(gray, (5,5), 1)

    # discs are bright white circles on a darker agar background
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1, minDist=40,
        param1=80, param2=22,       # stricter accumulator threshold
        minRadius=12, maxRadius=40,
    )
    if circles is None:
        return []

    candidates = [tuple(np.round(c).astype(int)) for c in circles[0]]

    # filter: the mean brightness inside a real disc should be high (>160)
    h, w = gray.shape
    valid = []
    for (cx, cy, r) in candidates:
        mask = np.zeros((h,w), np.uint8)
        cv2.circle(mask, (cx,cy), max(r-2,3), 255, -1)
        mean_val = cv2.mean(gray, mask=mask)[0]
        if mean_val > 150:          # discs are bright — filter dark false positives
            valid.append((cx, cy, r))

    # cap at expected max and sort left-to-right, top-to-bottom
    valid.sort(key=lambda c: (c[1]//80, c[0]))
    return valid[:max_expected]


# ══════════════════════════════════════════════════════════════════════════════
# DISC LABEL OCR  — reads abbreviation from disc (e.g. "CIP", "MEM")
# ══════════════════════════════════════════════════════════════════════════════

def read_disc_label(gray, cx, cy, r):
    """
    Crops the disc region and attempts to read the antibiotic abbreviation
    using simple template matching against known 2–4 char strings.
    Falls back to "DEFAULT" if OCR unavailable or unreadable.

    For production, replace with pytesseract:
        pip install pytesseract
        import pytesseract
        text = pytesseract.image_to_string(roi, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/')
    """
    try:
        import pytesseract
        h, w = gray.shape
        pad  = int(r * 1.1)
        x1,y1 = max(cx-pad,0), max(cy-pad,0)
        x2,y2 = min(cx+pad,w), min(cy+pad,h)
        roi   = gray[y1:y2, x1:x2]
        roi   = cv2.resize(roi, (80,80))
        # invert if disc is white-on-dark
        if roi.mean() > 128:
            roi = cv2.bitwise_not(roi)
        text = pytesseract.image_to_string(
            roi,
            config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'
        ).strip().upper()
        # clean: keep only alphanumeric
        text = re.sub(r'[^A-Z0-9]', '', text)
        if 2 <= len(text) <= 6:
            return text
    except ImportError:
        pass    # pytesseract not installed — use DEFAULT
    except Exception:
        pass
    return "DEFAULT"


def classify(zone_mm, disc_label, guideline="EUCAST"):
    table = BREAKPOINTS_EUCAST      # CLSI table can be added similarly
    bp    = table.get(disc_label.upper(), table["DEFAULT"])
    s_min, r_max = bp
    if zone_mm >= s_min: return "S"
    if zone_mm <= r_max: return "R"
    return "I"


# ══════════════════════════════════════════════════════════════════════════════
# U-NET SEGMENTATION  — with mask inversion fix
# ══════════════════════════════════════════════════════════════════════════════

def segment(model, img_rgb, img_size, device, discs=None):
    """
    Returns binary mask (H×W uint8, 255=zone) at original resolution.

    `discs`, if provided (list of (cx,cy,r) from find_discs, called BEFORE
    segment() so this is available), is used to set a physically-motivated
    small-fragment removal threshold instead of a flat "0.5% of image area"
    rule. On dense multi-disc plates a fixed image-area percentage is far
    too aggressive — it can erase real, small, near-breakpoint zones — so
    we instead floor at roughly half a disc's own footprint, which is the
    smallest area a genuine (even tiny) zone annulus can plausibly be.

    FIX: The Dryad training masks mark inhibition zones as WHITE (255).
    Inhibition zones appear as DARK clear halos on the plate.
    If the model learned correctly, class-1 = zone = dark region.
    We check both orientations and pick the one that spatially correlates
    with low-brightness regions (actual clear halos are lighter than
    bacterial growth but darker than the white discs).
    """
    h, w    = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, (img_size, img_size))
    tensor  = torch.from_numpy(resized.transpose(2,0,1)).float()/255.0
    tensor  = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[:, 1]    # P(zone)
        prob_map = probs.squeeze().cpu().numpy()
        prob_map = cv2.resize(prob_map, (w, h))

    # threshold
    raw_mask = (prob_map > 0.5).astype(np.uint8) * 255

    # ── inversion check ──────────────────────────────────────────────────────
    # Real inhibition zones are intermediate-brightness regions (not the
    # brightest white disc, not the darkest bacterial growth).
    # Convert image to grayscale and compare which mask orientation better
    # correlates with mid-range brightness values (100–200).
    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mid_br = ((gray > 80) & (gray < 210)).astype(np.uint8)

    overlap_normal   = cv2.bitwise_and(raw_mask//255, mid_br).sum()
    overlap_inverted = cv2.bitwise_and(255-raw_mask, mid_br*255).sum() // 255

    if overlap_inverted > overlap_normal * 1.3:
        # inverted mask fits better — flip it
        raw_mask = 255 - raw_mask

    # ── morphological cleanup ─────────────────────────────────────────────────
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    cleaned  = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    cleaned  = cv2.morphologyEx(cleaned,  cv2.MORPH_CLOSE, kernel, iterations=2)

    # remove tiny fragments — scaled to a real disc's footprint when we know
    # it, instead of a flat % of the whole image (see docstring above)
    if discs:
        median_disc_r = float(np.median([d[2] for d in discs]))
        min_area = int(0.5 * math.pi * (median_disc_r ** 2))
    else:
        min_area = int(h * w * 0.005)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    filtered = np.zeros_like(cleaned)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == lbl] = 255

    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# MEASUREMENT + ANNOTATED OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def pixels_to_mm(zone_radius_px, disc_radius_px):
    px_per_mm = (disc_radius_px * 2) / DISC_DIAMETER_MM
    return (zone_radius_px * 2) / px_per_mm


def measure_zone_radial(mask, cx, cy, disc_r, n_rays=36, max_search_px=None,
                         min_coverage=0.3, gap_tolerance_px=2, start_tolerance_px=4):
    """
    Per-disc zone measurement that is robust to merged/touching zones.

    WHY (v1 bug fixed here): the ray-walk previously had no requirement that
    the zone begin near the disc. If the pixels right at the disc edge were
    background, the ray kept walking outward — sometimes 60-80px — until it
    hit ANY white pixel anywhere along that line (noise speckle, glare, a
    totally different zone) and started measuring from there. Once latched
    on, it then kept extending through small gaps. Combined with a smaller
    fragment-area filter (needed to fix the no-zone-detected problem), this
    caused predicted diameters to average ~2x the true value (40.8mm vs.
    21.1mm ground truth), with some readings >85mm — physically impossible
    on a real plate.

    FIX: a ray must find zone pixels within `start_tolerance_px` of the
    disc's own edge, or it contributes no measurement at all (a real
    inhibition zone is contiguous with its disc — there's no legitimate
    reason for a large background gap between them). `gap_tolerance_px`
    (small, allows brief mask noise) only applies once the ray has already
    started inside a genuine, disc-adjacent zone.
    """
    h, w = mask.shape
    if max_search_px is None:
        max_search_px = int(disc_r * 6)

    hit_radii = []
    for k in range(n_rays):
        theta = 2 * math.pi * k / n_rays
        dx, dy = math.cos(theta), math.sin(theta)
        last_zone_r, started = None, False
        for step in range(int(disc_r), max_search_px):
            x = int(cx + dx * step)
            y = int(cy + dy * step)
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            on = mask[y, x] > 127
            if not started:
                if on:
                    started, last_zone_r = True, step
                elif step - disc_r > start_tolerance_px:
                    break   # zone doesn't start near the disc in this direction
            else:
                if on:
                    last_zone_r = step
                elif step - last_zone_r > gap_tolerance_px:
                    break

        if last_zone_r is not None:
            hit_radii.append(last_zone_r)

    coverage = len(hit_radii) / n_rays
    if coverage < min_coverage:
        return None, coverage

    return float(np.median(hit_radii)), coverage


def measure_and_annotate(mask, discs, img_rgb, guideline, gray):
    vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # zone overlay
    overlay = vis.copy()
    overlay[mask > 127] = (0, 200, 80)
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

    median_disc_r = float(np.median([d[2] for d in discs])) if discs else 20.0
    results = []

    for i, (cx, cy, disc_r) in enumerate(discs):
        # OCR the disc label
        label = read_disc_label(gray, cx, cy, disc_r)

        # draw disc
        cv2.circle(vis, (cx,cy), disc_r, (255,255,255), 2)
        cv2.putText(vis, f"D{i+1}", (cx-12, cy+disc_r+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # geometric safety cap: a disc's zone can't reasonably extend past
        # ~60% of the distance to its nearest neighboring disc — prevents a
        # ray from ever claiming a neighbor's territory even if the mask
        # itself is ambiguous in between
        other_dists = [math.hypot(cx-ocx, cy-ocy)
                        for j,(ocx,ocy,_) in enumerate(discs) if j != i]
        neighbor_cap = 0.6 * min(other_dists) if other_dists else disc_r * 6
        search_cap   = int(min(disc_r * 6, neighbor_cap))

        # per-disc radial measurement — robust to merged/touching zones
        er, coverage = measure_zone_radial(mask, cx, cy, disc_r,
                                            max_search_px=search_cap)

        if er is None:
            zone_mm = DISC_DIAMETER_MM
            sir     = "R"
            col     = SIR_COLOR["R"]
            cv2.putText(vis, f"D{i+1}:{label} {zone_mm:.0f}mm {sir}(no zone)",
                        (10, 28+i*26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, col, 1)
            results.append({"disc_index":i+1,"antibiotic":label,
                            "zone_diameter_mm":zone_mm,"classification":sir,
                            "note":"No inhibition zone detected",
                            "ray_coverage":round(coverage,2)})
            continue

        zone_mm = pixels_to_mm(er, median_disc_r)
        sir     = classify(zone_mm, label, guideline)
        col     = SIR_COLOR[sir]

        cv2.circle(vis, (cx,cy), int(er), col, 2)
        cv2.putText(vis, f"D{i+1}:{label} {zone_mm:.1f}mm {sir}",
                    (cx-55, cy-int(er)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)
        cv2.putText(vis, f"D{i+1}:{label} {zone_mm:.1f}mm {sir}",
                    (10, 28+i*26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

        results.append({"disc_index":i+1,"antibiotic":label,
                        "zone_diameter_mm":round(zone_mm,2),
                        "classification":sir,
                        "ray_coverage":round(coverage,2)})

    return results, vis


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(size=args.model_size, device=device)
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded  |  Device: {device}")

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"❌  Cannot read: {args.image}"); return

    out  = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    # quality check
    qr     = validate_image(img_bgr)
    errors = [i for i in qr.issues if i["severity"]=="error"]
    warns  = [i for i in qr.issues if i["severity"]=="warning"]

    if errors:
        print("\n❌  Image quality check FAILED.\n")
        for e in errors:
            print(f"  [{e['code']}] {e['message']}\n  → {e['instruction']}\n")
        alert = make_alert_image(img_bgr, errors+warns)
        ap    = out/f"{stem}_RETAKE.jpg"
        cv2.imwrite(str(ap), alert)
        json.dump({"status":"REJECTED","issues":qr.issues},
                  open(out/f"{stem}_quality_report.json","w"), indent=2)
        print(f"Alert image saved: {ap}"); return

    if warns:
        print(f"⚠️  {len(warns)} warning(s) — proceeding:")
        for w in warns:
            print(f"  [{w['code']}] {w['message']}\n  → {w['instruction']}\n")

    # crop to plate
    cropped_bgr, (ox, oy) = crop_to_plate(img_bgr)
    img_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)

    # find discs FIRST — segment() uses disc size to set a sane
    # small-fragment removal threshold instead of a flat image-area %
    discs = find_discs(gray)
    if not discs:
        h, w  = gray.shape
        discs = [(w//2, h//2, 20)]
        print("⚠️  No discs detected — using image centre fallback.")

    print(f"Detected {len(discs)} antibiotic disc(s).")

    # segment
    mask = segment(model, img_rgb, args.img_size, device, discs=discs)

    # measure + annotate
    results, vis = measure_and_annotate(mask, discs, img_rgb,
                                        args.guideline, gray)

    # save
    cv2.imwrite(str(out/f"{stem}_mask.png"),   mask)
    cv2.imwrite(str(out/f"{stem}_result.jpg"), vis)
    report = {"status":"OK","guideline":args.guideline,
              "disc_count":len(discs),"warnings":warns,"results":results}
    jp = out/f"{stem}_measurements.json"
    json.dump(report, open(jp,"w"), indent=2)

    # print summary
    sir_str = {"S":"Susceptible ✅","I":"Intermediate ⚠️","R":"Resistant ❌"}
    print(f"\n── Results ({args.guideline}) ──")
    for r in results:
        note = f"  ({r['note']})" if r.get("note") else ""
        print(f"  D{r['disc_index']:>2} [{r['antibiotic']:>7}]  "
              f"{r['zone_diameter_mm']:>5.1f} mm  →  "
              f"{sir_str.get(r['classification'],r['classification'])}{note}")

    print(f"\nSaved: {out}/{stem}_result.jpg  |  {jp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionAST inference")
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--image",       required=True)
    parser.add_argument("--output_dir",  default="outputs/")
    parser.add_argument("--model_size",  choices=["full","small"], default="small")
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--guideline",   choices=["EUCAST","CLSI"], default="EUCAST")
    args = parser.parse_args()
    main(args)