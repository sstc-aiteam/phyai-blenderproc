import blenderproc as bproc
import numpy as np
import os
import cv2
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SCENE_GLB_PATH  = "ward_nc.glb"        # room + embedded objects (no separate GLB)
TARGET_NAME     = "disposable_mask"    # matched case-insensitively against object names
OUTPUT_DIR      = "yolo_dataset_mask"
YOLO_CLASS_ID   = 0
BPROC_CAT_ID    = 1
CLASS_NAME      = "disposable_mask"

NUM_SAMPLES     = 90
IMAGE_WIDTH     = 640
IMAGE_HEIGHT    = 640
RENDER_SAMPLES  = 128

# Camera: 3 distances × (3 elevations × 8 azimuths + 6 near-zenith) = 3 × 30 = 90 poses
ELEVATIONS_DEG  = [20, 35, 55]
NUM_AZIMUTHS    = 8
TOP_DOWN_EL_DEG = 75
TOP_DOWN_COUNT  = 6
DIST_FACTORS    = [1.2, 2.0, 3.5]   # very close, mid, and far orbit rings

VAL_RATIO       = 0.10
MIN_MASK_PIXELS = 500    # disposable mask may be small; lower threshold than trashcan


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def object_bounds(objs):
    corners = np.vstack([o.get_bound_box() for o in objs])
    lo, hi = corners.min(0), corners.max(0)
    return (lo + hi) / 2.0, float(np.linalg.norm(hi - lo))


def make_cam_pose(cam_pos: np.ndarray, target: np.ndarray):
    fwd = target - cam_pos
    rot = bproc.camera.rotation_from_forward_vec(fwd, inplane_rot=0.0)
    return bproc.math.build_transformation_mat(cam_pos.tolist(), rot)


def yolo_bbox(seg: np.ndarray, cat_id: int, W: int, H: int):
    mask = seg == cat_id
    if mask.sum() < MIN_MASK_PIXELS:
        return None
    ys, xs = np.where(mask)
    cx = (xs.min() + xs.max()) / 2.0 / W
    cy = (ys.min() + ys.max()) / 2.0 / H
    bw = (xs.max() - xs.min()) / W
    bh = (ys.max() - ys.min()) / H
    return float(cx), float(cy), float(bw), float(bh)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    bproc.init()

    for split in ("train", "val"):
        Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)

    # ── Load scene (room + all embedded objects) ──────────────────────────────
    print(f"[INFO] Loading scene: {SCENE_GLB_PATH}")
    all_objs = bproc.loader.load_obj(os.path.abspath(SCENE_GLB_PATH))
    if not all_objs:
        raise RuntimeError(f"Failed to load {SCENE_GLB_PATH}")
    print(f"[INFO] Loaded {len(all_objs)} objects total")
    print(f"[INFO] Object names: {[o.get_name() for o in all_objs]}")

    # ── Find target by name ───────────────────────────────────────────────────
    mask_objs = [
        o for o in all_objs
        if TARGET_NAME.lower() in o.get_name().lower()
    ]
    if not mask_objs:
        raise RuntimeError(
            f"No object matching '{TARGET_NAME}' found in {SCENE_GLB_PATH}.\n"
            f"Available names: {[o.get_name() for o in all_objs]}"
        )
    print(f"[INFO] Found {len(mask_objs)} target mesh(es): "
          f"{[o.get_name() for o in mask_objs]}")

    for obj in mask_objs:
        obj.set_cp("category_id", BPROC_CAT_ID)

    # ── Target centre + orbit radius ──────────────────────────────────────────
    center, diag = object_bounds(mask_objs)
    print(f"[INFO] Target centre={np.round(center, 3)}  diagonal={diag:.4f} m")
    if diag < 1e-4:
        raise RuntimeError("Target object has zero size — check GLB contents")

    print(f"[INFO] Orbit radii: {[round(diag * f, 4) for f in DIST_FACTORS]} m")

    # ── Lighting ──────────────────────────────────────────────────────────────
    # Energies kept low (≤100 W) so the scene is visible without overexposure.
    for offset, energy in [
        ([ 2.0, -2.0,  3.0], 80),
        ([-2.0,  2.0,  2.5], 60),
        ([-1.5, -1.5,  3.5], 50),
        ([ 1.5,  1.5,  1.5], 40),
    ]:
        lt = bproc.types.Light()
        lt.set_type("POINT")
        lt.set_location([
            center[0] + offset[0] * diag,
            center[1] + offset[1] * diag,
            center[2] + offset[2] * diag,
        ])
        lt.set_energy(energy)
        lt.set_radius(0.5)

    # ── Camera intrinsics ─────────────────────────────────────────────────────
    bproc.camera.set_resolution(IMAGE_WIDTH, IMAGE_HEIGHT)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=35,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        lens_unit="MILLIMETERS",
    )

    # ── Camera poses: 2 distances × (3 el × 8 az + 6 near-zenith) = 60 total ──
    azimuths = np.linspace(0, 2 * np.pi, NUM_AZIMUTHS, endpoint=False)
    el_top   = np.radians(TOP_DOWN_EL_DEG)
    for dist_factor in DIST_FACTORS:
        orbit_r = diag * dist_factor
        for el_deg in ELEVATIONS_DEG:
            el = np.radians(el_deg)
            for az in azimuths:
                pos = center + orbit_r * np.array([
                    np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el),
                ])
                bproc.camera.add_camera_pose(make_cam_pose(pos, center))

        for az in np.linspace(0, 2 * np.pi, TOP_DOWN_COUNT, endpoint=False):
            pos = center + orbit_r * np.array([
                np.cos(el_top) * np.cos(az),
                np.cos(el_top) * np.sin(az),
                np.sin(el_top),
            ])
            bproc.camera.add_camera_pose(make_cam_pose(pos, center))

    poses_per_dist = len(ELEVATIONS_DEG) * NUM_AZIMUTHS + TOP_DOWN_COUNT
    total_poses    = len(DIST_FACTORS) * poses_per_dist
    print(f"[INFO] Camera poses registered: {total_poses} ({len(DIST_FACTORS)} distances × {poses_per_dist})")

    # ── Render ────────────────────────────────────────────────────────────────
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id"],
        default_values={"category_id": 0},
    )
    bproc.renderer.set_max_amount_of_samples(RENDER_SAMPLES)

    print("[INFO] Rendering …")
    data     = bproc.renderer.render()
    colors   = data["colors"]
    seg_maps = data["category_id_segmaps"]
    print(f"[INFO] Rendered {len(colors)} frames")

    # ── Train / val split (deterministic) ────────────────────────────────────
    rng           = np.random.default_rng(seed=42)
    n_val         = max(1, int(NUM_SAMPLES * VAL_RATIO))
    val_positions = set(rng.choice(NUM_SAMPLES, n_val, replace=False).tolist())

    # ── Save images + labels ──────────────────────────────────────────────────
    saved       = {"train": 0, "val": 0}
    skipped     = 0
    saved_total = 0

    for img, seg in zip(colors, seg_maps):
        if saved_total >= NUM_SAMPLES:
            break

        bbox = yolo_bbox(seg, BPROC_CAT_ID, IMAGE_WIDTH, IMAGE_HEIGHT)
        if bbox is None:
            skipped += 1
            continue

        split = "val" if saved_total in val_positions else "train"
        stem  = f"{saved_total:06d}"

        cv2.imwrite(
            f"{OUTPUT_DIR}/images/{split}/{stem}.jpg",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        cx, cy, bw, bh = bbox
        with open(f"{OUTPUT_DIR}/labels/{split}/{stem}.txt", "w") as f:
            f.write(f"{YOLO_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        saved[split] += 1
        saved_total  += 1

    # ── dataset.yaml ──────────────────────────────────────────────────────────
    yaml_path = f"{OUTPUT_DIR}/dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"# Auto-generated YOLO dataset — {CLASS_NAME}\n")
        f.write(f"path: {os.path.abspath(OUTPUT_DIR)}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n\n")
        f.write("nc: 1\n")
        f.write(f"names: ['{CLASS_NAME}']\n")

    print(f"\n[DONE] train={saved['train']}  val={saved['val']}  skipped={skipped}")
    print(f"[DONE] dataset  → {os.path.abspath(OUTPUT_DIR)}")
    print(f"[DONE] YOLO cfg → {yaml_path}")


if __name__ == "__main__":
    main()
