"""
Visualize to verify the YOLO dataset bounding boxes and labels.
Usage:
    python visualize_yolo_dataset.py                                 # random 16 images, show window
    python visualize_yolo_dataset.py --split val                     # val split only
    python visualize_yolo_dataset.py --n 32                          # show 32 images
    python visualize_yolo_dataset.py --save yolo_dataset_vis_out     # save to folder instead of showing
    python visualize_yolo_dataset.py --all                           # save every image
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

DATASET_DIR = Path(__file__).parent / "yolo_dataset"
VISUALIZE_OUT_DIR = Path(__file__).parent / "yolo_dataset_vis_out"
TEXT_COLOR  = (255, 255, 255)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.55
THICKNESS   = 2

# Distinct per-class colors (BGR)
CLASS_COLORS = [
    (0, 255, 0),
    (255, 80, 80),
    (80, 80, 255),
    (0, 220, 220),
    (220, 0, 220),
    (220, 220, 0),
    (255, 140, 0),
    (0, 165, 255),
]


def class_color(cls_id: int) -> tuple[int, int, int]:
    return CLASS_COLORS[cls_id % len(CLASS_COLORS)]


def load_dataset_meta(dataset_dir: Path) -> tuple[list[str], str]:
    """Return (class_names, task) from dataset.yaml. task defaults to 'detect'."""
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("names", []), data.get("task", "detect")


def draw_annotations(img: np.ndarray, label_path: Path,
                     class_names: list[str], task: str) -> np.ndarray:
    H, W = img.shape[:2]
    img = img.copy()

    if not label_path.exists():
        return img

    with open(label_path) as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        color  = class_color(cls_id)
        label  = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        if task == "segment" and len(parts) > 5:
            coords = list(map(float, parts[1:]))
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= W
            pts[:, 1] *= H
            pts = pts.astype(np.int32)

            # Semi-transparent filled mask
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

            # Polygon outline
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=THICKNESS)

            # Label at top of bounding rect
            x, y, bw, bh = cv2.boundingRect(pts)
            _draw_label(img, label, x, y, color)

        else:
            # Bounding-box detection format
            if len(parts) != 5:
                continue
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)
            _draw_label(img, label, x1, y1, color)

    return img


def _draw_label(img: np.ndarray, label: str, x: int, y: int,
                color: tuple[int, int, int]) -> None:
    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
    ty = max(y - 4, th + 4)
    cv2.rectangle(img, (x, ty - th - 4), (x + tw + 4, ty + baseline), color, -1)
    cv2.putText(img, label, (x + 2, ty), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)


def collect_pairs(dataset_dir: Path, split: str | None) -> list[tuple[Path, Path]]:
    splits = [split] if split else ["train", "val"]
    pairs = []
    for s in splits:
        img_dir = dataset_dir / "images" / s
        lbl_dir = dataset_dir / "labels" / s
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            pairs.append((img_path, lbl_path))
    return pairs


def make_grid(images: list[np.ndarray], cols: int = 4) -> np.ndarray:
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(img, (w, h))
    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO dataset annotations")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--split",   choices=["train", "val"], default=None)
    parser.add_argument("--n",       type=int, default=16, help="number of images in grid")
    parser.add_argument("--save",    type=Path, nargs="?", const=VISUALIZE_OUT_DIR,
                        default=None, help="output directory (default: yolo_dataset_vis_out)")
    parser.add_argument("--all",     action="store_true", help="process and save every image")
    parser.add_argument("--cols",    type=int, default=4, help="grid columns")
    parser.add_argument("--width",   type=int, default=3840,
                        help="max output grid width in pixels (default: 3840)")
    args = parser.parse_args()

    class_names, task = load_dataset_meta(args.dataset)
    print(f"Task:    {task}")
    print(f"Classes: {class_names}")

    pairs = collect_pairs(args.dataset, args.split)
    print(f"Found {len(pairs)} images")

    if not pairs:
        print("No images found — check dataset path and split.")
        return

    if args.all or args.save:
        save_dir = args.save or args.dataset / "visualized"
        save_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in pairs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            vis = draw_annotations(img, lbl_path, class_names, task)
            out = save_dir / f"{img_path.parent.name}_{img_path.name}"
            cv2.imwrite(str(out), vis)

        print(f"Saved {len(pairs)} visualized images to {save_dir}/")
        return

    # Grid preview
    sample = random.sample(pairs, min(args.n, len(pairs)))
    annotated = []
    for img_path, lbl_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        annotated.append(draw_annotations(img, lbl_path, class_names, task))

    grid = make_grid(annotated, cols=args.cols)

    max_w = args.width
    if grid.shape[1] > max_w:
        scale = max_w / grid.shape[1]
        grid = cv2.resize(grid, (max_w, int(grid.shape[0] * scale)))

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if has_display:
        cv2.imshow("YOLO Dataset Verification", grid)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        out_path = VISUALIZE_OUT_DIR / "grid_preview.jpg"
        VISUALIZE_OUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"No display detected — saved grid to {out_path}")


if __name__ == "__main__":
    main()
