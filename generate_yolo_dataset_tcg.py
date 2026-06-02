import blenderproc as bproc
import bpy
import numpy as np
import os
import cv2
from pathlib import Path
from mathutils import Vector

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
GLB_PATH       = "trashcan_clothbag_green.glb"
#ROOM_GLB_PATH  = "r207_meeting_room.glb"
ROOM_GLB_PATH  = "fake_ward.glb"
OUTPUT_DIR     = "yolo_dataset"
YOLO_CLASS_ID  = 0
BPROC_CAT_ID   = 1
CLASS_NAME     = "trashcan_clothbag_green"

NUM_SAMPLES    = 300
IMAGE_WIDTH    = 640
IMAGE_HEIGHT   = 640
RENDER_SAMPLES = 128
TRASHCAN_SCALE = 4.0

# Camera: 7 elevations × 48 azimuths = 336  +  24 near-zenith = 360 total poses
# (~20 % buffer above NUM_SAMPLES so skipped frames don't drop us below 300)
ELEVATIONS_DEG  = [10, 20, 30, 40, 50, 60, 70]
NUM_AZIMUTHS    = 48
TOP_DOWN_EL_DEG = 80
TOP_DOWN_COUNT  = 24
DIST_FACTOR     = 3.0

VAL_RATIO       = 0.10
MIN_MASK_PIXELS = 800

# Collision-avoidance tuning
WALL_MARGIN    = 0.30   # ray-cast: min distance between trashcan edge and wall surface
FURNITURE_GAP  = 0.40   # AABB: min XY separation from furniture bounding boxes
RAY_HEIGHTS    = [0.25, 0.70]   # heights above floor_z at which horizontal rays are cast
N_WALL_RAYS    = 12             # number of horizontal rays per height level


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
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
# Ray casting helpers  (wall / floor mesh-baked objects)
# ──────────────────────────────────────────────────────────────────────────────

def _ray_cast(origin, direction):
    """Thin wrapper around Blender's scene ray_cast. Returns (hit, location, …)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return bpy.context.scene.ray_cast(depsgraph, Vector(origin), Vector(direction))


def find_floor_z_by_raycast(x: float, y: float, start_z: float = 10.0, stop_z: float = None):
    """
    Cast a ray straight down from (x, y, start_z), piercing through each surface
    encountered (ceiling, roof slabs, furniture tops), and return the Z of the
    lowest surface hit above stop_z.  This ensures we find the interior floor
    rather than the exterior ceiling that a single cast from above would return.
    Falls back to None if no geometry is hit at all.
    """
    current_z = start_z
    lowest_hit = None

    for _ in range(20):          # max surface-piercing iterations
        hit, loc, _, _, _, _ = _ray_cast([x, y, current_z], [0.0, 0.0, -1.0])
        if not hit:
            break
        lowest_hit = float(loc.z)
        current_z = lowest_hit - 0.05   # step 5 cm past this surface
        if stop_z is not None and current_z < stop_z:
            break

    return lowest_hit


def check_wall_clearance(cx: float, cy: float, floor_z: float, tc_radius: float) -> bool:
    """
    Cast N_WALL_RAYS horizontal rays at RAY_HEIGHTS above floor_z.
    Returns True only when every direction has at least (tc_radius + WALL_MARGIN) clearance.

    This catches walls, door frames, and any large baked-mesh surface regardless
    of their AABB size, which AABB tests alone cannot handle correctly.
    """
    required = tc_radius + WALL_MARGIN
    angles = np.linspace(0.0, 2.0 * np.pi, N_WALL_RAYS, endpoint=False)

    for h in RAY_HEIGHTS:
        origin = [cx, cy, floor_z + h]
        for angle in angles:
            direction = [float(np.cos(angle)), float(np.sin(angle)), 0.0]
            hit, loc, _, _, _, _ = _ray_cast(origin, direction)
            if hit:
                dist = float((Vector(loc) - Vector(origin)).length)
                if dist < required:
                    return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# AABB helpers  (isolated / furniture objects)
# ──────────────────────────────────────────────────────────────────────────────

def _furniture_aabbs(room_objs, floor_z: float, room_lo, room_hi):
    """
    Return (lo, hi) AABBs for objects classified as furniture — i.e. objects that
    rise above the floor but whose XY footprint does NOT span >80 % of the room in
    either axis.  Large planar surfaces (walls, floor slab, ceiling) are excluded
    here because their over-sized AABBs would block every candidate; they are
    handled instead by the ray-casting pass.
    """
    aabbs = []
    room_xy = np.array([room_hi[0] - room_lo[0], room_hi[1] - room_lo[1]])
    for obj in room_objs:
        corners = np.array(obj.get_bound_box())
        lo, hi = corners.min(0), corners.max(0)
        if hi[2] <= floor_z + 0.05:        # flat-on-floor slab → skip
            continue
        obj_xy = np.array([hi[0] - lo[0], hi[1] - lo[1]])
        if (obj_xy / room_xy > 0.80).any():  # wall / ceiling → skip (ray cast handles it)
            continue
        aabbs.append((lo, hi))
    return aabbs


# ──────────────────────────────────────────────────────────────────────────────
# Two-tier collision-free floor placement
# ──────────────────────────────────────────────────────────────────────────────

def find_clear_floor_position(tc_objs, room_objs, room_lo, room_hi, floor_z):
    """
    Grid-search the room floor for an XY position that passes:
      1. AABB test  — trashcan footprint does not overlap any furniture AABB
                      (with FURNITURE_GAP safety margin)
      2. Ray-cast test — horizontal rays confirm ≥ (tc_radius + WALL_MARGIN)
                         clearance from all wall/mesh-baked surfaces

    Candidates are sorted by distance from the room centre so the most
    central free position is preferred.  Falls back to room centre if nothing
    passes both tests.
    """
    tc_corners = np.vstack([o.get_bound_box() for o in tc_objs])
    tc_lo  = tc_corners.min(0)
    tc_hi  = tc_corners.max(0)
    tc_ctr = (tc_lo + tc_hi) / 2.0
    tc_r   = max(tc_hi[0] - tc_lo[0], tc_hi[1] - tc_lo[1]) / 2.0

    room_ctr = (room_lo + room_hi) / 2.0
    dz       = floor_z - tc_lo[2]

    margin   = tc_r + 0.15
    x_lo_s, x_hi_s = room_lo[0] + margin, room_hi[0] - margin
    y_lo_s, y_hi_s = room_lo[1] + margin, room_hi[1] - margin

    furniture = _furniture_aabbs(room_objs, floor_z, room_lo, room_hi)
    print(f"[INFO] Furniture AABBs for AABB test: {len(furniture)}")

    step = max(0.20, tc_r)
    candidates = []
    x = x_lo_s
    while x <= x_hi_s + 1e-6:
        y = y_lo_s
        while y <= y_hi_s + 1e-6:
            d2 = (x - room_ctr[0]) ** 2 + (y - room_ctr[1]) ** 2
            candidates.append((d2, x, y))
            y += step
        x += step
    candidates.sort()
    print(f"[INFO] Candidate grid points: {len(candidates)}")

    for _, cx, cy in candidates:
        dx, dy = cx - tc_ctr[0], cy - tc_ctr[1]
        p_lo   = tc_lo + [dx, dy, dz]
        p_hi   = tc_hi + [dx, dy, dz]

        # ── Tier 1: AABB test for furniture/isolated objects ─────────────────
        if any(
            (p_lo[0] < r_hi[0] - FURNITURE_GAP) and (r_lo[0] + FURNITURE_GAP < p_hi[0]) and
            (p_lo[1] < r_hi[1] - FURNITURE_GAP) and (r_lo[1] + FURNITURE_GAP < p_hi[1])
            for r_lo, r_hi in furniture
        ):
            continue

        # ── Tier 2: Ray-cast test for wall/floor mesh-baked surfaces ─────────
        if check_wall_clearance(cx, cy, floor_z, tc_r):
            print(f"[INFO] Collision-free spot at XY=({cx:.3f}, {cy:.3f})")
            return np.array([dx, dy, dz])

    print("[WARN] No collision-free spot found; falling back to room centre")
    return np.array([room_ctr[0] - tc_ctr[0], room_ctr[1] - tc_ctr[1], dz])


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    bproc.init()

    for split in ("train", "val"):
        Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)

    # ── Load room ─────────────────────────────────────────────────────────────
    print(f"[INFO] Loading room: {ROOM_GLB_PATH}")
    room_objs = bproc.loader.load_obj(os.path.abspath(ROOM_GLB_PATH))
    if not room_objs:
        raise RuntimeError(f"Failed to load {ROOM_GLB_PATH}")

    room_corners = np.vstack([o.get_bound_box() for o in room_objs])
    room_lo  = room_corners.min(0)
    room_hi  = room_corners.max(0)
    room_ctr = (room_lo + room_hi) / 2.0
    print(f"[INFO] Room bounds lo={np.round(room_lo, 2)}  hi={np.round(room_hi, 2)}")

    # Refine floor_z: pierce downward through ceiling/roof until the interior floor.
    # stop_z = room_lo[2] prevents the ray from going below the model's bottom face.
    raycasted_floor = find_floor_z_by_raycast(
        room_ctr[0], room_ctr[1],
        start_z=room_hi[2] + 0.5,
        stop_z=room_lo[2],
    )
    floor_z = raycasted_floor if raycasted_floor is not None else float(room_lo[2])
    print(f"[INFO] Floor Z (ray-cast refined): {floor_z:.4f}")

    # ── Load trashcan ─────────────────────────────────────────────────────────
    print(f"[INFO] Loading object: {GLB_PATH}")
    tc_objs = bproc.loader.load_obj(os.path.abspath(GLB_PATH))
    if not tc_objs:
        raise RuntimeError(f"Failed to load {GLB_PATH}")
    for obj in tc_objs:
        obj.set_cp("category_id", BPROC_CAT_ID)

    for obj in tc_objs:
        obj.set_scale([TRASHCAN_SCALE] * 3)
    print(f"[INFO] Trashcan scaled × {TRASHCAN_SCALE}")

    # ── Collision-free placement ──────────────────────────────────────────────
    delta = find_clear_floor_position(tc_objs, room_objs, room_lo, room_hi, floor_z)
    for obj in tc_objs:
        obj.set_location((np.array(obj.get_location()) + delta).tolist())

    center, diag = object_bounds(tc_objs)
    print(f"[INFO] Trashcan centre={np.round(center, 3)}  diagonal={diag:.4f} m")

    # ── Lighting ──────────────────────────────────────────────────────────────
    # Use AREA lights (large source = soft penumbra) instead of POINT lights.
    # Size is in metres; larger = softer shadows.
    light_configs = [
        ([ 2.5, -2.5,  3.5], 800, 2.0),
        ([-2.5,  2.5,  3.0], 600, 2.0),
        ([-2.0, -2.0,  4.0], 500, 1.5),
        ([ 2.0,  2.0,  2.0], 400, 1.5),
    ]
    for offset, energy, size in light_configs:
        lt = bproc.types.Light()
        lt.set_type("POINT")
        lt.set_location([center[0] + offset[0] * diag,
                         center[1] + offset[1] * diag,
                         center[2] + offset[2] * diag])
        lt.set_radius(size)
        lt.set_energy(energy)

    # Soft fill from above: wide area light acts like a diffuse sky panel.
    ceil_lt = bproc.types.Light()
    ceil_lt.set_type("AREA")
    ceil_lt.set_location([center[0], center[1], floor_z + (room_hi[2] - floor_z) * 0.85])
    ceil_lt.set_energy(1600)
    ceil_lt.set_rotation_euler([0, 0, 1])

    # ── Camera intrinsics ─────────────────────────────────────────────────────
    bproc.camera.set_resolution(IMAGE_WIDTH, IMAGE_HEIGHT)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=35,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        lens_unit="MILLIMETERS",
    )

    # ── Camera poses ──────────────────────────────────────────────────────────
    # Orbit radius: clamped to stay inside room walls
    room_xy_r = min(room_hi[0] - room_ctr[0], room_hi[1] - room_ctr[1]) * 0.85
    orbit_r   = min(diag * DIST_FACTOR, room_xy_r)

    # Regular elevation rings: ELEVATIONS_DEG × NUM_AZIMUTHS
    azimuths = np.linspace(0, 2 * np.pi, NUM_AZIMUTHS, endpoint=False)
    for el_deg in ELEVATIONS_DEG:
        el = np.radians(el_deg)
        for az in azimuths:
            pos = center + orbit_r * np.array([
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el),
            ])
            bproc.camera.add_camera_pose(make_cam_pose(pos, center))

    # Near-zenith ring
    top_azimuths = np.linspace(0, 2 * np.pi, TOP_DOWN_COUNT, endpoint=False)
    el_top = np.radians(TOP_DOWN_EL_DEG)
    for az in top_azimuths:
        pos = center + orbit_r * np.array([
            np.cos(el_top) * np.cos(az),
            np.cos(el_top) * np.sin(az),
            np.sin(el_top),
        ])
        bproc.camera.add_camera_pose(make_cam_pose(pos, center))

    total_poses = len(ELEVATIONS_DEG) * NUM_AZIMUTHS + TOP_DOWN_COUNT
    print(f"[INFO] Camera poses registered: {total_poses}")

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

    # ── dataset.yaml ─────────────────────────────────────────────────────────
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
