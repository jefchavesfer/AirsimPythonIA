"""
split_by_sequence.py - split the generated dataset into train/val/test BY SEQUENCE.

Frames from the same sequence are near-duplicates, so a random per-frame split
leaks information into validation. This script keeps whole sequences together,
stratified by class, and can hold out an entire Unreal map as the test set.

Output layout (YOLO, same as earlier prepareTrainVal.py):
    <dst>/images/{train,val,test}/*.png|jpg
    <dst>/labels/{train,val,test}/*.txt      class xc yc w h (normalized)
    <dst>/data.yaml
Optional extra formats for SSD-style pipelines (--formats yolo coco voc):
    <dst>/annotations/instances_{split}.json  COCO (category ids start at 1; 0 = background)
    <dst>/labels_voc/{split}/*.xml            Pascal VOC (pixel boxes)

Usage
  python split_by_sequence.py --src dataset --modality rgb --dst yoloTrainVal
  python split_by_sequence.py --src dataset --modality ir  --dst yoloTrainVal_ir --formats yolo coco voc
  python split_by_sequence.py --src dataset --dst yoloTrainVal --test-map CityMap
"""
import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


def read_boxes(label_path):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append((int(parts[0]), *[float(x) for x in parts[1:]]))
    return boxes


def write_voc(path, fname, w, h, boxes, names):
    objs = ""
    for cid, xc, yc, bw, bh in boxes:
        x0, y0 = int(round((xc - bw / 2) * w)), int(round((yc - bh / 2) * h))
        x1, y1 = int(round((xc + bw / 2) * w)), int(round((yc + bh / 2) * h))
        objs += (f"  <object><name>{names[cid]}</name><pose>Unspecified</pose><truncated>0</truncated>"
                 f"<difficult>0</difficult><bndbox><xmin>{max(0, x0)}</xmin><ymin>{max(0, y0)}</ymin>"
                 f"<xmax>{min(w, x1)}</xmax><ymax>{min(h, y1)}</ymax></bndbox></object>\n")
    path.write_text(f"<annotation>\n  <filename>{fname}</filename>\n"
                    f"  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>\n"
                    f"  <segmented>0</segmented>\n{objs}</annotation>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset root containing metadata.csv")
    ap.add_argument("--modality", default="rgb", help="subfolder to split (rgb or ir)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--test-map", default=None, help="send every sequence of this map to the test split")
    ap.add_argument("--formats", nargs="+", choices=["yolo", "coco", "voc"], default=["yolo"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    img_dir = src / args.modality / "images"
    lab_dir = src / args.modality / "labels"

    seqs = defaultdict(lambda: {"files": [], "cid": None, "map": None})
    names = {}
    with open(src / "metadata.csv", newline="") as f:
        for r in csv.DictReader(f):
            s = seqs[int(r["seq"])]
            s["files"].append((r["file"], r.get("size_bin", "none")))
            s["cid"] = int(r["class_id"])
            s["map"] = r["map"]
            if int(r["class_id"]) >= 0:
                names[int(r["class_id"])] = r["class_name"]

    rng = random.Random(args.seed)
    assign = {}
    by_class = defaultdict(list)
    for sid, s in seqs.items():
        if args.test_map and s["map"] == args.test_map:
            assign[sid] = "test"
        else:
            by_class[s["cid"]].append(sid)
    for cid, ids in by_class.items():
        rng.shuffle(ids)
        n_test = 0 if args.test_map else round(len(ids) * args.test)
        n_val = round(len(ids) * args.val)
        for i, sid in enumerate(ids):
            assign[sid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        if "voc" in args.formats:
            (dst / "labels_voc" / split).mkdir(parents=True, exist_ok=True)
    if "coco" in args.formats:
        (dst / "annotations").mkdir(parents=True, exist_ok=True)

    stats = defaultdict(lambda: defaultdict(int))
    coco = {sp: {"images": [], "annotations": [],
                 "categories": [{"id": c + 1, "name": n} for c, n in sorted(names.items())]}
            for sp in ("train", "val", "test")}
    img_size = None
    for sid, s in seqs.items():
        split = assign[sid]
        for fname, sbin in s["files"]:
            stem = Path(fname).stem
            src_img, src_lab = img_dir / fname, lab_dir / f"{stem}.txt"
            if not src_img.exists() or not src_lab.exists():
                print(f"[WARN] missing pair for {fname}")
                continue
            shutil.copy2(src_img, dst / "images" / split / fname)
            shutil.copy2(src_lab, dst / "labels" / split / f"{stem}.txt")
            boxes = read_boxes(src_lab)
            stats[split]["images"] += 1
            stats[split]["instances"] += len(boxes)
            if not boxes:
                stats[split]["background"] += 1
            for _ in boxes:
                stats[split]["bin_" + sbin] += 1
            if ("coco" in args.formats or "voc" in args.formats):
                if img_size is None:
                    im = cv2.imread(str(src_img))
                    img_size = (im.shape[1], im.shape[0])
                w, h = img_size
                if "voc" in args.formats:
                    write_voc(dst / "labels_voc" / split / f"{stem}.xml", fname, w, h, boxes, names)
                if "coco" in args.formats:
                    iid = len(coco[split]["images"]) + 1
                    coco[split]["images"].append({"id": iid, "file_name": fname, "width": w, "height": h})
                    for cid, xc, yc, bw, bh in boxes:
                        coco[split]["annotations"].append({
                            "id": len(coco[split]["annotations"]) + 1, "image_id": iid, "category_id": cid + 1,
                            "bbox": [round((xc - bw / 2) * w, 2), round((yc - bh / 2) * h, 2),
                                     round(bw * w, 2), round(bh * h, 2)],
                            "area": round(bw * w * bh * h, 2), "iscrowd": 0})

    if "coco" in args.formats:
        for sp, data in coco.items():
            (dst / "annotations" / f"instances_{sp}.json").write_text(json.dumps(data))

    with open(dst / "data.yaml", "w") as f:
        f.write(f"path: {dst.resolve().as_posix()}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\nnames:\n")
        for cid in sorted(names):
            f.write(f"  {cid}: {names[cid]}\n")

    print(f"{'split':6s} {'sequences':>9s} {'images':>7s} {'instances':>9s} {'background':>10s}   "
          f"{'small':>6s} {'medium':>6s} {'large':>6s}  (size bins at SSD-300 resolution)")
    for split in ("train", "val", "test"):
        n_seq = sum(1 for v in assign.values() if v == split)
        st = stats[split]
        print(f"{split:6s} {n_seq:9d} {st['images']:7d} {st['instances']:9d} {st['background']:10d}   "
              f"{st['bin_small']:6d} {st['bin_medium']:6d} {st['bin_large']:6d}")
    print(f"\nWrote {dst / 'data.yaml'}  (formats: {', '.join(args.formats)})")


if __name__ == "__main__":
    main()
