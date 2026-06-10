import blenderproc as bproc
import bpy
import argparse
import numpy as np
import os
import cv2
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SCENE_GLB_PATH = "ward_nc.glb"        # room + embedded objects (no separate GLB)
OUTPUT_DIR     = "yolo_dataset"

# (name_pattern, yolo_class_id, bproc_cat_id, apply_z_flip)
TARGETS = [
    ("disposable_mask",         0, 1, True),
    ("waterproof_bandages_ppb", 1, 2, True),
    ("ac_remotecontrol",        2, 3, True),
    ("syringe_nipro",           3, 4, True),
    ("cotton_swabs_ppb",        4, 5, True),
    ("paracetamol",             5, 6, True),
    ("bottle_alcohol_spray",    6, 7, False),
]

CLASS_NAMES = [t[0] for t in TARGETS]

IMAGE_WIDTH     = 640
IMAGE_HEIGHT    = 640
RENDER_SAMPLES  = 128

# Camera: 3 distances × (3 elevations × 6 azimuths + 6 near-zenith) = 3 × 24 = 82 poses per orientation
ELEVATIONS_DEG  = [30, 50, 70]
NUM_AZIMUTHS    = 6
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


def yolo_seg(seg: np.ndarray, cat_id: int, W: int, H: int):
    mask = (seg == cat_id).astype(np.uint8) * 255
    if int(mask.sum() // 255) < MIN_MASK_PIXELS:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_MASK_PIXELS:
        return None
    pts = contour.reshape(-1, 2).astype(float)
    pts[:, 0] /= W
    pts[:, 1] /= H
    return pts.flatten().tolist()


def register_camera_poses(center: np.ndarray, diag: float):
    """Add one full orbit ring of camera poses around `center`."""
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


def reset_camera_keyframes():
    """Clear all camera pose keyframes so the next pass starts at frame 1."""
    cam = bpy.context.scene.camera
    if cam and cam.animation_data:
        cam.animation_data_clear()
    bpy.context.scene.frame_end = 0


def apply_z_flip(mask_objs, floor_z: float):
    """Rotate each mask mesh 180° around its world X-axis and restore Z floor.

    The rotation is applied purely to the orientation (translation preserved),
    then the object is shifted up so its lowest vertex stays at floor_z,
    keeping the mask fully above the chair surface.
    """
    import mathutils
    rot_flip = mathutils.Matrix.Rotation(np.pi, 4, 'X')
    for obj in mask_objs:
        bl_obj = obj.blender_obj
        old_mat = bl_obj.matrix_world.copy()
        loc     = old_mat.translation.copy()
        new_rot = rot_flip.to_3x3() @ old_mat.to_3x3()
        bl_obj.matrix_world = mathutils.Matrix.Translation(loc) @ new_rot.to_4x4()
        corners  = np.array(obj.get_bound_box())
        new_floor = float(corners[:, 2].min())
        bl_obj.location.z += floor_z - new_floor + 10e-3  # tiny offset to prevent Z-fighting


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["detect", "segment"], default="detect",
                        help="detect: bounding-box labels; segment: polygon-mask labels")
    args, _ = parser.parse_known_args()
    task = args.task
    print(f"[INFO] Task: {task}")

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

    # ── Find and configure all target objects ─────────────────────────────────
    target_data = {}
    for tname, yolo_id, cat_id, do_z_flip in TARGETS:
        objs = [o for o in all_objs if tname.lower() in o.get_name().lower()]
        if not objs:
            raise RuntimeError(
                f"No object matching '{tname}' found in {SCENE_GLB_PATH}.\n"
                f"Available names: {[o.get_name() for o in all_objs]}"
            )
        print(f"[INFO] '{tname}': found {len(objs)} mesh(es): {[o.get_name() for o in objs]}")
        for obj in objs:
            obj.set_cp("category_id", cat_id)
        center, diag = object_bounds(objs)
        corners_init = np.vstack([o.get_bound_box() for o in objs])
        floor_z      = float(corners_init[:, 2].min())
        print(f"[INFO] '{tname}': centre={np.round(center, 3)}  diagonal={diag:.4f} m")
        print(f"[INFO] '{tname}': orbit radii={[round(diag * f, 4) for f in DIST_FACTORS]} m")
        if diag < 1e-4:
            raise RuntimeError(f"Target '{tname}' has zero size — check GLB contents")
        target_data[tname] = {
            "objs": objs, "yolo_id": yolo_id, "cat_id": cat_id,
            "do_z_flip": do_z_flip, "center": center, "diag": diag, "floor_z": floor_z,
        }

    # ── Lighting (centered on first target) ──────────────────────────────────
    first        = target_data[TARGETS[0][0]]
    center0, diag0 = first["center"], first["diag"]
    for offset, energy in [
        ([ 2.0, -2.0,  3.0], 60),
        ([-2.0,  2.0,  2.5], 40),
        ([-1.5, -1.5,  3.5], 30),
        ([ 1.5,  1.5,  1.5], 20),
    ]:
        lt = bproc.types.Light()
        lt.set_type("POINT")
        lt.set_location([
            center0[0] + offset[0] * diag0,
            center0[1] + offset[1] * diag0,
            center0[2] + offset[2] * diag0,
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

    # ── Segmentation + render settings ───────────────────────────────────────
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id"],
        default_values={"category_id": 0},
    )
    bproc.renderer.set_max_amount_of_samples(RENDER_SAMPLES)

    poses_per_pass = len(DIST_FACTORS) * (len(ELEVATIONS_DEG) * NUM_AZIMUTHS + TOP_DOWN_COUNT)
    print(f"[INFO] Camera poses per pass: {poses_per_pass}")

    # ── Rendering passes: orbit each target; z_flip targets get a second pass ─
    all_colors: list = []
    all_segs:   list = []
    pass_num = 0

    for tname, yolo_id, cat_id, do_z_flip in TARGETS:
        td     = target_data[tname]
        objs   = td["objs"]
        center = td["center"]
        diag   = td["diag"]

        passes = ["original", "Z-flipped"] if do_z_flip else ["original"]
        for pass_idx, label in enumerate(passes):
            if pass_idx == 1:
                # Z-flip second pass: clear poses, flip object, recompute centre
                reset_camera_keyframes()
                apply_z_flip(objs, td["floor_z"])
                center, _ = object_bounds(objs)
                print(f"[INFO] '{tname}' Z-flipped centre: {np.round(center, 3)}")
            elif pass_num > 0:
                # New target's first pass: clear poses from previous target/pass
                reset_camera_keyframes()

            register_camera_poses(center, diag)
            print(f"[INFO] Rendering '{tname}' ({label}) …")
            data = bproc.renderer.render()
            all_colors.extend(data["colors"])
            all_segs.extend(data["category_id_segmaps"])
            print(f"[INFO] '{tname}' ({label}): {len(data['colors'])} frames")
            pass_num += 1

    print(f"[INFO] Total raw frames: {len(all_colors)}")

    # ── Filter frames with at least one valid annotation ──────────────────────
    valid_frames = []
    for img, seg in zip(all_colors, all_segs):
        annotations = []
        for tname, yolo_id, cat_id, _ in TARGETS:
            if task == "segment":
                pts = yolo_seg(seg, cat_id, IMAGE_WIDTH, IMAGE_HEIGHT)
                if pts is not None:
                    annotations.append((yolo_id, pts))
            else:
                bbox = yolo_bbox(seg, cat_id, IMAGE_WIDTH, IMAGE_HEIGHT)
                if bbox is not None:
                    cx, cy, bw, bh = bbox
                    annotations.append((yolo_id, cx, cy, bw, bh))
        if annotations:
            valid_frames.append((img, annotations))

    print(f"[INFO] Valid frames (≥1 annotation): {len(valid_frames)}")

    # ── Train / val split (deterministic) ────────────────────────────────────
    n_total = len(valid_frames)
    rng     = np.random.default_rng(seed=42)
    n_val   = max(1, int(n_total * VAL_RATIO))
    val_idx = set(rng.choice(n_total, n_val, replace=False).tolist())

    # ── Save images + labels ──────────────────────────────────────────────────
    saved = {"train": 0, "val": 0}
    for i, (img, annotations) in enumerate(valid_frames):
        split = "val" if i in val_idx else "train"
        stem  = f"{i:06d}"

        cv2.imwrite(
            f"{OUTPUT_DIR}/images/{split}/{stem}.jpg",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        with open(f"{OUTPUT_DIR}/labels/{split}/{stem}.txt", "w") as f:
            for ann in annotations:
                if task == "segment":
                    yolo_id, pts = ann
                    coords = " ".join(f"{v:.6f}" for v in pts)
                    f.write(f"{yolo_id} {coords}\n")
                else:
                    yolo_id, cx, cy, bw, bh = ann
                    f.write(f"{yolo_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        saved[split] += 1

    # ── dataset.yaml ──────────────────────────────────────────────────────────
    yaml_path = f"{OUTPUT_DIR}/dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"# Auto-generated YOLO dataset — {', '.join(CLASS_NAMES)}\n")
        f.write(f"task: {task}\n")
        f.write(f"path: {os.path.abspath(OUTPUT_DIR)}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    skipped = len(all_colors) - len(valid_frames)
    print(f"\n[DONE] train={saved['train']}  val={saved['val']}  skipped={skipped}")
    print(f"[DONE] dataset  → {os.path.abspath(OUTPUT_DIR)}")
    print(f"[DONE] YOLO cfg → {yaml_path}")


if __name__ == "__main__":
    main()
