"""
Visualize to verify the YOLO dataset bounding boxes and labels.
Usage:
    python visualize_yolo_dataset.py                                 # random 16 images, show window
    python visualize_yolo_dataset.py --split val                     # val split only
    python visualize_yolo_dataset.py --n 32                          # show 32 images
    python visualize_yolo_dataset.py --save yolo_dataset_vis_out # save to folder instead of showing
    python visualize_yolo_dataset.py --all                           # save every image
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

DATASET_DIR = Path(__file__).parent / "yolo_dataset"
VISUALIZE_OUT_DIR = Path(__file__).parent / "yolo_dataset_vis_out"
BOX_COLOR   = (0, 255, 0)   # green
TEXT_COLOR  = (255, 255, 255)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.55
THICKNESS   = 2


def load_class_names(dataset_dir: Path) -> list[str]:
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("names", [])


def draw_boxes(img: np.ndarray, label_path: Path, class_names: list[str]) -> np.ndarray:
    H, W = img.shape[:2]
    img = img.copy()

    if not label_path.exists():
        return img

    with open(label_path) as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])

        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)

        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), BOX_COLOR, -1)
        cv2.putText(img, label, (x1 + 2, ty), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)

    return img


def collect_pairs(dataset_dir: Path, split: str | None) -> list[tuple[Path, Path]]:
    splits = [split] if split else ["train", "val"]
    pairs = []
    for s in splits:
        img_dir   = dataset_dir / "images" / s
        lbl_dir   = dataset_dir / "labels" / s
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
    parser = argparse.ArgumentParser(description="Visualize YOLO dataset bounding boxes")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--split",   choices=["train", "val"], default=None)
    parser.add_argument("--n",       type=int, default=16, help="number of images in grid")
    parser.add_argument("--save",    type=Path, nargs="?", const=VISUALIZE_OUT_DIR, default=None, help="output directory (default: yolo_dataset_vis_out)")
    parser.add_argument("--all",     action="store_true", help="process and save every image")
    parser.add_argument("--cols",    type=int, default=4, help="grid columns")
    args = parser.parse_args()

    class_names = load_class_names(args.dataset)
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
            vis = draw_boxes(img, lbl_path, class_names)
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
        annotated.append(draw_boxes(img, lbl_path, class_names))

    grid = make_grid(annotated, cols=args.cols)

    # Scale down if grid is too wide for the screen
    max_w = 1600
    if grid.shape[1] > max_w:
        scale = max_w / grid.shape[1]
        grid = cv2.resize(grid, (max_w, int(grid.shape[0] * scale)))

    cv2.imshow("YOLO Dataset Verification", grid)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
