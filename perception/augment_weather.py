"""Synthetic adverse-condition augmentation (night, rain, fog).

Used two ways:
  1. to build an *adverse* copy of a validation set, so detection accuracy
     (mAP) can be compared between normal and adverse conditions;
  2. to enlarge the training set so the detector becomes robust to them.

    python -m perception.augment_weather --images data/val/images --labels data/val/labels \
        --out data/val_adverse --mode mixed
Labels are copied unchanged (the geometry does not change).
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def night(img, rng):
    gamma = rng.uniform(2.0, 3.0)
    out = ((img / 255.0) ** gamma * 255).astype(np.uint8)
    noise = rng.normal(0, 6, img.shape)
    out = np.clip(out * np.array([1.0, 0.95, 0.85]) + noise, 0, 255).astype(np.uint8)
    return out


def rain(img, rng):
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.uint8)
    for _ in range(int(h * w / 600)):
        x, y = rng.integers(0, w), rng.integers(0, h)
        length = rng.integers(10, 25)
        cv2.line(layer, (x, y), (x + length // 4, y + length), 200, 1)
    layer = cv2.blur(layer, (3, 3))
    out = cv2.addWeighted(img, 0.8, cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR), 0.5, 0)
    return (out * 0.85).astype(np.uint8)


def fog(img, rng):
    strength = rng.uniform(0.35, 0.6)
    fogged = img * (1 - strength) + 220 * strength
    return cv2.GaussianBlur(fogged.astype(np.uint8), (5, 5), 0)


MODES = {"night": night, "rain": rain, "fog": fog}


def augment(img, mode, rng):
    if mode == "mixed":
        mode = rng.choice(list(MODES))
    return MODES[mode](img, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="mixed", choices=["mixed", *MODES])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out_i, out_l = Path(a.out) / "images", Path(a.out) / "labels"
    out_i.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(a.images).iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(out_i / p.name), augment(img, a.mode, rng))
        lab = Path(a.labels) / (p.stem + ".txt")
        if lab.exists():
            shutil.copy(lab, out_l / lab.name)
        n += 1
    print(f"wrote {n} {a.mode} images to {a.out}")


if __name__ == "__main__":
    main()
