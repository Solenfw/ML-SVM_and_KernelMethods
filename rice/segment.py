from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

RAW_DIR = Path("data/raw")
META_PATH = Path("data/metadata.csv")
OVERLAY_DIR = Path("data/overlays")
CROP_DIR = Path("data/crops")
OUT_CSV = Path("data/segmented_grains.csv")

CROP_TOP = 0.0
CROP_BOTTOM = 0.74
CROP_LEFT = 0.0
CROP_RIGHT = 1.0

OTSU_SCALE = 0.90
MIN_AREA_MM2 = 3.5
MAX_AREA_MM2 = 28.0


def save_grain(roi_bgr, overlay, crop_mask, bbox, centroid, prop, img_stem, idx, mm_per_pixel):
    minr, minc, maxr, maxc = bbox
    grain_id = f"{img_stem}_{idx:03d}"
    crop_bgr = roi_bgr[minr:maxr, minc:maxc]

    cv2.imwrite(str(CROP_DIR / f"{grain_id}.png"), crop_bgr)
    cv2.imwrite(str(CROP_DIR / f"{grain_id}_mask.png"), crop_mask)

    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shifted = [c + np.array([[[minc, minr]]], dtype=c.dtype) for c in contours]
    cv2.drawContours(overlay, shifted, -1, (0, 255, 0), 2)
    cv2.putText(
        overlay,
        str(idx),
        (int(centroid[1]), int(centroid[0])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )

    return {
        "grain_id": grain_id,
        "image_id": img_stem,
        "area_mm2": prop.area * (mm_per_pixel ** 2),
        "perimeter_mm": prop.perimeter * mm_per_pixel,
        "major_axis_mm": prop.major_axis_length * mm_per_pixel,
        "minor_axis_mm": prop.minor_axis_length * mm_per_pixel,
    }


def draw_rejected(overlay, crop_mask, bbox):
    minr, minc, _, _ = bbox
    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shifted = [c + np.array([[[minc, minr]]], dtype=c.dtype) for c in contours]
    cv2.drawContours(overlay, shifted, -1, (0, 255, 255), 1)


def segment_image(img_path: Path, mm_per_pixel: float, stats: dict):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return []

    h_full, w_full = img_bgr.shape[:2]
    roi_bgr = img_bgr[
        int(h_full * CROP_TOP) : int(h_full * CROP_BOTTOM),
        int(w_full * CROP_LEFT) : int(w_full * CROP_RIGHT),
    ]

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    otsu_t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(blur, int(otsu_t * OTSU_SCALE), 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_bin = ndi.binary_fill_holes(binary > 0)

    cc_labels, _ = ndi.label(mask_bin)
    h, w = gray.shape
    cc_props = regionprops(cc_labels)

    px_per_mm2 = 1.0 / (mm_per_pixel ** 2)
    min_px = MIN_AREA_MM2 * px_per_mm2
    max_px = MAX_AREA_MM2 * px_per_mm2

    candidate_areas = [p.area for p in cc_props if min_px <= p.area <= max_px]
    if not candidate_areas:
        return []
    median_area = float(np.median(candidate_areas))

    overlay = roi_bgr.copy()
    valid_grains = []
    idx = 0

    sigma = max(1.0, 0.18 / mm_per_pixel)
    min_dist = max(10, int(1.3 / mm_per_pixel))
    thresh_abs = max(2.5, 0.25 / mm_per_pixel)

    for p in cc_props:
        minr, minc, maxr, maxc = p.bbox
        if minr <= 2 or minc <= 2 or maxr >= h - 2 or maxc >= w - 2:
            stats["border"] += 1
            continue

        if p.area < 0.45 * median_area:
            stats["small"] += 1
            if p.area >= 0.20 * median_area:
                draw_rejected(overlay, (p.image.astype(np.uint8)) * 255, p.bbox)
            continue

        if p.area <= 1.75 * median_area and p.solidity >= 0.75:
            idx += 1
            crop_mask = (p.image.astype(np.uint8)) * 255
            valid_grains.append(
                save_grain(roi_bgr, overlay, crop_mask, p.bbox, p.centroid, p, img_path.stem, idx, mm_per_pixel)
            )
            continue

        sub_mask = np.pad(p.image, 1, mode="constant", constant_values=False)
        dist = ndi.distance_transform_edt(sub_mask)
        dist_smooth = ndi.gaussian_filter(dist, sigma=sigma)
        coords = peak_local_max(
            dist_smooth,
            min_distance=min_dist,
            threshold_abs=thresh_abs,
            labels=sub_mask,
        )

        if len(coords) < 2:
            stats["clump_unsplit"] += 1
            draw_rejected(overlay, (p.image.astype(np.uint8)) * 255, p.bbox)
            continue

        mask_peaks = np.zeros(dist.shape, dtype=bool)
        mask_peaks[tuple(coords.T)] = True
        markers, _ = ndi.label(mask_peaks)
        sub_labels = watershed(-dist_smooth, markers, mask=sub_mask)

        for sp in regionprops(sub_labels):
            s_minr, s_minc, s_maxr, s_maxc = sp.bbox
            g_bbox = (minr + s_minr - 1, minc + s_minc - 1, minr + s_maxr - 1, minc + s_maxc - 1)
            crop_mask = (sp.image.astype(np.uint8)) * 255

            if sp.area < 0.45 * median_area:
                stats["small"] += 1
                draw_rejected(overlay, crop_mask, g_bbox)
                continue
            if sp.area > 1.75 * median_area:
                stats["large"] += 1
                draw_rejected(overlay, crop_mask, g_bbox)
                continue
            if sp.solidity < 0.72:
                stats["solidity"] += 1
                draw_rejected(overlay, crop_mask, g_bbox)
                continue

            idx += 1
            g_centroid = (minr + sp.centroid[0] - 1, minc + sp.centroid[1] - 1)
            valid_grains.append(
                save_grain(roi_bgr, overlay, crop_mask, g_bbox, g_centroid, sp, img_path.stem, idx, mm_per_pixel)
            )

    cv2.imwrite(str(OVERLAY_DIR / f"{img_path.stem}_overlay.jpg"), overlay)
    return valid_grains


def main():
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_PATH)
    all_rows = []
    stats = {"border": 0, "small": 0, "large": 0, "solidity": 0, "clump_unsplit": 0}

    for _, row in meta.iterrows():
        img_path = RAW_DIR / str(row["file"])
        grains = segment_image(img_path, float(row["mm_per_pixel"]), stats)
        for g in grains:
            g["variety"] = row["variety"]
            g["session"] = row["session"]
            all_rows.append(g)

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f"Da tach xong {len(all_rows)} hat hop le.")
    print(f"Thong ke bo qua: {stats}")


if __name__ == "__main__":
    main()