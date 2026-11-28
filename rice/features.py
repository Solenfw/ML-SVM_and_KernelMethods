from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops

SEG_CSV = Path("data/segmented_grains.csv")
CROP_DIR = Path("data/crops")
OUT_CSV = Path("data/features.csv")

GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]


def extract_one(row):
    grain_id = row["grain_id"]
    crop_bgr = cv2.imread(str(CROP_DIR / f"{grain_id}.png"))
    mask = cv2.imread(str(CROP_DIR / f"{grain_id}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if crop_bgr is None or mask is None:
        return None

    m = mask > 0
    if m.sum() == 0:
        return None

    props = regionprops(m.astype(np.uint8))
    if not props:
        return None
    p = props[0]

    area_mm2 = float(row["area_mm2"])
    perimeter_mm = float(row["perimeter_mm"])
    major_mm = float(row["major_axis_mm"])
    minor_mm = float(row["minor_axis_mm"])
    aspect_ratio = major_mm / minor_mm if minor_mm > 0 else 0.0
    circularity = (4.0 * np.pi * area_mm2) / (perimeter_mm ** 2) if perimeter_mm > 0 else 0.0
    solidity = float(p.solidity)

    feat = {
        "grain_id": grain_id,
        "image_id": row["image_id"],
        "session": row["session"],
        "variety": row["variety"],
        "area_mm2": area_mm2,
        "perimeter_mm": perimeter_mm,
        "major_axis_mm": major_mm,
        "minor_axis_mm": minor_mm,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "solidity": solidity,
    }

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)

    spaces = [
        ("rgb", rgb, ["R", "G", "B"]),
        ("hsv", hsv, ["H", "S", "V"]),
        ("lab", lab, ["L", "A", "B"]),
    ]
    for prefix, img_space, ch_names in spaces:
        for i, ch in enumerate(ch_names):
            vals = img_space[:, :, i][m].astype(np.float32)
            feat[f"{prefix}_{ch}_mean"] = float(vals.mean())
            feat[f"{prefix}_{ch}_std"] = float(vals.std())

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_q = (gray // 4).astype(np.uint8)
    gray_q[~m] = 0

    glcm = graycomatrix(
        gray_q,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )
    for d_idx, d in enumerate([1, 2]):
        for prop_name in GLCM_PROPS:
            val = graycoprops(glcm, prop_name)[d_idx, :].mean()
            feat[f"glcm_d{d}_{prop_name}"] = float(val)

    return feat


def main():
    df_seg = pd.read_csv(SEG_CSV)
    rows = []
    for _, row in df_seg.iterrows():
        item = extract_one(row)
        if item is not None:
            rows.append(item)

    df_feat = pd.DataFrame(rows)
    df_feat.to_csv(OUT_CSV, index=False)
    print(f"Da trich xuat {len(df_feat)} hat x {df_feat.shape[1] - 4} dac trung -> {OUT_CSV}")


if __name__ == "__main__":
    main()