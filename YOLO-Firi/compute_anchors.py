"""
Re-estimate YOLO-FIRI anchors (12 = 4 detection scales x 3) from the training labels.

Usage (from the YOLO-FIRI repo root):
  python compute_anchors.py --labels C:/Users/jefch/AirSimPython/yoloTrainVal_ir/labels/train \
         --img-size 1280 --write models/yolov5-FIRI-3cls.yaml

Anchors are sorted by area; the 3 smallest go to P2/4, the next 3 to P3/8, etc.
Background frames (empty .txt) are ignored. Boxes are clipped to >= 2 px like in YOLOv5.
"""

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans

LEVELS = ["P2/4", "P3/8", "P4/16", "P5/32"]


def load_wh(label_dir, img_w, img_h, img_size):
    """Box sizes (w, h) in pixels after YOLOv5's letterbox resize (long side -> img_size)."""
    r = img_size / max(img_w, img_h)
    wh, n_files, n_empty = [], 0, 0
    for f in sorted(Path(label_dir).glob("*.txt")):
        n_files += 1
        rows = [ln.split() for ln in f.read_text().strip().splitlines() if ln.strip()]
        if not rows:
            n_empty += 1
            continue
        a = np.array(rows, dtype=np.float64)
        wh.append(a[:, 3:5] * [img_w * r, img_h * r])
    if not wh:
        raise SystemExit(f"no boxes found in {label_dir}")
    return np.concatenate(wh, 0), n_files, n_empty


def metric(k, wh, thr):
    """YOLOv5 anchor metric: ratio-based best-possible-recall style fitness."""
    r = wh[:, None] / k[None]
    x = np.minimum(r, 1 / r).min(2)           # (n_boxes, n_anchors) worst-side ratio
    best = x.max(1)
    fitness = (best * (best > 1 / thr)).mean()
    bpr = (best > 1 / thr).mean()
    aat = (x > 1 / thr).sum(1).mean()          # anchors above threshold per box
    return fitness, bpr, aat


def kmean_anchors(wh, n=12, thr=4.0, gen=1000, seed=0):
    rng = np.random.default_rng(seed)
    s = wh.std(0)
    k, _ = kmeans(wh / s, n, iter=30)
    k = k * s
    assert len(k) == n, f"k-means returned {len(k)} centres, expected {n}"
    k = k[np.argsort(k.prod(1))]
    f = metric(k, wh, thr)[0]
    sh, mp, S = k.shape, 0.9, 0.1
    for _ in range(gen):                        # genetic refinement
        v = np.ones(sh)
        while (v == 1).all():
            v = ((rng.random(sh) < mp) * rng.random() * rng.standard_normal(sh) * S + 1).clip(0.3, 3.0)
        kg = (k * v).clip(min=2.0)
        fg = metric(kg, wh, thr)[0]
        if fg > f:
            f, k = fg, kg.copy()
    return k[np.argsort(k.prod(1))]


def format_block(k, per_level=3):
    lines = []
    for i, lvl in enumerate(LEVELS):
        trio = k[i * per_level:(i + 1) * per_level]
        s = ", ".join(f"{int(round(w))},{int(round(h))}" for w, h in trio)
        lines.append(f"  - [{s}]  # {lvl}")
    return "anchors:\n" + "\n".join(lines) + "\n"


def write_yaml(path, block):
    text = Path(path).read_text()
    new, n = re.subn(r"anchors:\n(?:\s*- \[.*\].*\n)+", block, text, count=1)
    if n != 1:
        raise SystemExit(f"could not find the anchors block in {path}")
    Path(path).write_text(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="folder with YOLO .txt labels of the TRAIN split")
    ap.add_argument("--img-size", type=int, default=1280, help="training image size (long side)")
    ap.add_argument("--img-w", type=int, default=1280)
    ap.add_argument("--img-h", type=int, default=720)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--thr", type=float, default=4.0, help="anchor_t of the hyp file")
    ap.add_argument("--gen", type=int, default=1000)
    ap.add_argument("--write", default="", help="model yaml whose anchors block should be replaced")
    a = ap.parse_args()

    wh, n_files, n_empty = load_wh(a.labels, a.img_w, a.img_h, a.img_size)
    print(f"{n_files} label files ({n_empty} empty/background), {len(wh)} boxes")
    print(f"box size at {a.img_size}px: min {wh.min(0).round(1)}, median {np.median(wh, 0).round(1)}, "
          f"max {wh.max(0).round(1)}")
    wh = wh[(wh >= 2.0).all(1)]

    k = kmean_anchors(wh, a.n, a.thr, a.gen)
    f, bpr, aat = metric(k, wh, a.thr)
    print(f"anchors: fitness {f:.3f}, best possible recall {bpr:.4f}, anchors>thr per box {aat:.2f}")
    block = format_block(k, a.n // len(LEVELS))
    print(block)
    if a.write:
        write_yaml(a.write, block)
        print(f"anchors written to {a.write}")


if __name__ == "__main__":
    main()
