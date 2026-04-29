import blenderproc as bproc
# BlenderProc YOLO dataset generator for trashcan_clothbag_green.glb
# Usage: blenderproc run generate_yolo_dataset.py
# Output: yolo_dataset/{images,labels}/{train,val}/ + dataset.yaml
import numpy as np
import os
import cv2
from pathlib import Path

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
GLB_PATH          = "trashcan_clothbag_green.glb"
OUTPUT_DIR        = "yolo_dataset"
YOLO_CLASS_ID     = 0          # class index written into .txt labels
BPROC_CAT_ID      = 1          # category_id assigned to the object inside BlenderProc
CLASS_NAME        = "trashcan_clothbag_green"

IMAGE_WIDTH       = 640
IMAGE_HEIGHT      = 640
RENDER_SAMPLES    = 64         # lower = faster; raise to 128 for cleaner images

# Camera grid
ELEVATIONS_DEG    = [10, 25, 40, 55, 70]     # 5 elevation bands
NUM_AZIMUTHS      = 24                        # every 15°  → 24 directions
DIST_FACTORS      = [2.2, 3.2, 4.5]          # multiples of object diagonal
TOP_DOWN_EL_DEG   = 85                        # near-zenith shots
TOP_DOWN_COUNT    = 12                        # azimuth steps for top-down

VAL_RATIO         = 0.10

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def object_bounds(objs):
    """(center, diagonal) over all mesh objects."""
    corners = np.vstack([o.get_bound_box() for o in objs])
    lo, hi  = corners.min(0), corners.max(0)
    return (lo + hi) / 2.0, float(np.linalg.norm(hi - lo))


def make_cam_pose(cam_pos, target):
    fwd = np.asarray(target) - np.asarray(cam_pos)
    rot = bproc.camera.rotation_from_forward_vec(fwd, inplane_rot=0.0)
    return bproc.math.build_transformation_mat(cam_pos.tolist(), rot)


def yolo_bbox_from_catmap(seg, cat_id, W, H):
    """
    Return (cx, cy, w, h) normalised to [0,1] for category cat_id,
    or None when the object is not visible.
    """
    mask = (seg == cat_id)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    cx = (xs.min() + xs.max()) / 2.0 / W
    cy = (ys.min() + ys.max()) / 2.0 / H
    bw = (xs.max() - xs.min()) / W
    bh = (ys.max() - ys.min()) / H
    return float(cx), float(cy), float(bw), float(bh)

# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    bproc.init()

    # Output directories
    for split in ("train", "val"):
        Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)

    # ── Load object ────────────────────────────────────────
    print(f"[INFO] Loading {GLB_PATH} …")
    objs = bproc.loader.load_obj(os.path.abspath(GLB_PATH))
    if not objs:
        raise RuntimeError(f"No objects loaded from {GLB_PATH}")
    for obj in objs:
        obj.set_cp("category_id", BPROC_CAT_ID)

    center, diag = object_bounds(objs)
    print(f"[INFO] center={np.round(center,3)}, diagonal={diag:.4f}")

    # ── Lighting ───────────────────────────────────────────
    # Four point lights at different positions for even coverage
    light_configs = [
        ([ diag*2.5, -diag*2.5,  diag*3.5],  600),
        ([-diag*2.5,  diag*2.5,  diag*3.0],  500),
        ([-diag*2.0, -diag*2.0,  diag*4.0],  400),
        ([ diag*2.0,  diag*2.0,  diag*2.0],  350),
    ]
    for loc, energy in light_configs:
        lt = bproc.types.Light()
        lt.set_type("POINT")
        lt.set_location([center[0] + loc[0], center[1] + loc[1], center[2] + loc[2]])
        lt.set_energy(energy)

    # Ambient sun so the back-faces are not completely black
    sun = bproc.types.Light()
    sun.set_type("SUN")
    sun.set_energy(1.5)
    sun.set_rotation_euler([0.5, 0.3, 0.0])

    # ── Camera intrinsics ──────────────────────────────────
    bproc.camera.set_resolution(IMAGE_WIDTH, IMAGE_HEIGHT)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=35,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        lens_unit="MILLIMETERS",
    )

    # ── Camera poses ───────────────────────────────────────
    azimuths = np.linspace(0, 360, NUM_AZIMUTHS, endpoint=False)

    # Grid: distance × elevation × azimuth
    for dist_f in DIST_FACTORS:
        dist = diag * dist_f
        for el_deg in ELEVATIONS_DEG:
            el = np.radians(el_deg)
            for az_deg in azimuths:
                az  = np.radians(az_deg)
                pos = np.array([
                    center[0] + dist * np.cos(el) * np.cos(az),
                    center[1] + dist * np.cos(el) * np.sin(az),
                    center[2] + dist * np.sin(el),
                ])
                bproc.camera.add_camera_pose(make_cam_pose(pos, center))

    # Top-down shots
    top_azimuths = np.linspace(0, 360, TOP_DOWN_COUNT, endpoint=False)
    el_top = np.radians(TOP_DOWN_EL_DEG)
    for dist_f in DIST_FACTORS:
        dist = diag * dist_f
        for az_deg in top_azimuths:
            az  = np.radians(az_deg)
            pos = np.array([
                center[0] + dist * np.cos(el_top) * np.cos(az),
                center[1] + dist * np.cos(el_top) * np.sin(az),
                center[2] + dist * np.sin(el_top),
            ])
            bproc.camera.add_camera_pose(make_cam_pose(pos, center))

    # ── Render ─────────────────────────────────────────────
    bproc.renderer.enable_segmentation_output(map_by=["category_id"])
    bproc.renderer.set_max_amount_of_samples(RENDER_SAMPLES)

    print("[INFO] Rendering …")
    data = bproc.renderer.render()

    colors   = data["colors"]                     # list[H×W×3 uint8]
    seg_maps = data["category_id_segmaps"]        # list[H×W  int]

    total = len(colors)
    print(f"[INFO] Rendered {total} frames")

    # ── Train / val split ──────────────────────────────────
    rng  = np.random.default_rng(seed=42)
    perm = rng.permutation(total)
    n_val = max(1, int(total * VAL_RATIO))
    val_set = set(perm[:n_val].tolist())

    # ── Save images + labels ───────────────────────────────
    saved   = {"train": 0, "val": 0}
    skipped = 0

    for i, (img, seg) in enumerate(zip(colors, seg_maps)):
        bbox = yolo_bbox_from_catmap(seg, BPROC_CAT_ID, IMAGE_WIDTH, IMAGE_HEIGHT)
        if bbox is None:
            skipped += 1
            continue

        split = "val" if i in val_set else "train"
        stem  = f"{i:06d}"

        cv2.imwrite(
            f"{OUTPUT_DIR}/images/{split}/{stem}.jpg",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        cx, cy, bw, bh = bbox
        with open(f"{OUTPUT_DIR}/labels/{split}/{stem}.txt", "w") as f:
            f.write(f"{YOLO_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        saved[split] += 1

    # ── dataset.yaml ───────────────────────────────────────
    yaml_path = f"{OUTPUT_DIR}/dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"# Auto-generated YOLO dataset — {CLASS_NAME}\n")
        f.write(f"path: {os.path.abspath(OUTPUT_DIR)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['{CLASS_NAME}']\n")

    print(f"\n[DONE]  train={saved['train']}  val={saved['val']}  skipped={skipped}")
    print(f"[DONE]  dataset → {os.path.abspath(OUTPUT_DIR)}")
    print(f"[DONE]  YOLO config → {yaml_path}")


if __name__ == "__main__":
    main()
