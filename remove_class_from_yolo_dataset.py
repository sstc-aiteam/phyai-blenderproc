"""
Remove a class from a YOLO dataset directory.
- Strips all annotations of the target class from label files
- Remaps remaining class IDs to fill the gap
- Updates data.yaml (nc and names)
- Optionally deletes image+label files that become empty after removal
"""

import argparse
import shutil
from pathlib import Path

import yaml


def remove_class(dataset_dir: str, class_name: str, delete_empty: bool = False):
    dataset = Path(dataset_dir)
    yaml_path = dataset / "data.yaml"

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    names: list[str] = cfg["names"]
    if class_name not in names:
        print(f"Class '{class_name}' not found in {yaml_path}. Names: {names}")
        return

    target_id = names.index(class_name)
    print(f"Removing class '{class_name}' (id={target_id}) from {dataset_dir}")

    splits = [d for d in (dataset / "train", dataset / "valid", dataset / "test") if d.exists()]

    total_files = 0
    total_removed_lines = 0
    total_deleted_files = 0

    for split in splits:
        labels_dir = split / "labels"
        images_dir = split / "images"
        if not labels_dir.exists():
            continue

        for label_file in sorted(labels_dir.glob("*.txt")):
            lines = label_file.read_text().splitlines()
            kept = []
            removed = 0
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                cid = int(parts[0])
                if cid == target_id:
                    removed += 1
                    continue
                # remap: shift ids above the removed class down by 1
                new_cid = cid if cid < target_id else cid - 1
                kept.append(f"{new_cid} " + " ".join(parts[1:]))

            total_files += 1
            total_removed_lines += removed

            if not kept and delete_empty:
                # remove label file and corresponding image
                label_file.unlink()
                stem = label_file.stem
                for img_ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    img = images_dir / (stem + img_ext)
                    if img.exists():
                        img.unlink()
                        break
                total_deleted_files += 1
            else:
                label_file.write_text("\n".join(kept) + ("\n" if kept else ""))

    # update data.yaml
    new_names = [n for n in names if n != class_name]
    cfg["names"] = new_names
    cfg["nc"] = len(new_names)
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)

    print(f"Done.")
    print(f"  Label files processed : {total_files}")
    print(f"  Annotation lines removed: {total_removed_lines}")
    if delete_empty:
        print(f"  Empty image/label pairs deleted: {total_deleted_files}")
    print(f"  Updated {yaml_path}")
    print(f"  New classes ({len(new_names)}): {new_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove a class from a YOLO dataset.")
    parser.add_argument("dataset_dir", help="Path to the dataset directory containing data.yaml")
    parser.add_argument("class_name", help="Class name to remove")
    parser.add_argument(
        "--delete-empty",
        action="store_true",
        help="Delete image+label files that have no annotations left after removal",
    )
    args = parser.parse_args()
    remove_class(args.dataset_dir, args.class_name, delete_empty=args.delete_empty)
