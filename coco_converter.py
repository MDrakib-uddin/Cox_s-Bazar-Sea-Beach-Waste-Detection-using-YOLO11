from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml


def load_classes(dataset_dir: Path) -> tuple[dict[int, int], list[str]]:
    with open(dataset_dir / "train" / "_annotations.coco.json", encoding="utf-8") as file:
        coco = json.load(file)
    used_ids = {annotation["category_id"] for annotation in coco["annotations"]}
    categories = [
        category
        for category in sorted(coco["categories"], key=lambda item: item["id"])
        if category["id"] in used_ids
    ]
    category_map = {category["id"]: index for index, category in enumerate(categories)}
    return category_map, [category["name"] for category in categories]


def convert_split(coco_dir: Path, output_dir: Path, output_split: str, category_map: dict[int, int]) -> tuple[int, int]:
    with open(coco_dir / "_annotations.coco.json", encoding="utf-8") as file:
        coco = json.load(file)
    image_dir = output_dir / "images" / output_split
    label_dir = output_dir / "labels" / output_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    image_count = 0
    box_count = 0
    for image in coco["images"]:
        source = coco_dir / image["file_name"]
        if not source.exists():
            raise FileNotFoundError(f"Missing image: {source}")
        destination = image_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        image_width, image_height = float(image["width"]), float(image["height"])
        label_lines = []
        for annotation in annotations_by_image.get(image["id"], []):
            category_id = annotation["category_id"]
            if category_id not in category_map:
                continue
            x, y, width, height = map(float, annotation["bbox"])
            x1, y1 = max(0.0, min(x, image_width)), max(0.0, min(y, image_height))
            x2, y2 = max(0.0, min(x + width, image_width)), max(0.0, min(y + height, image_height))
            if x2 <= x1 or y2 <= y1:
                continue
            values = (category_map[category_id], (x1 + x2) / 2 / image_width, (y1 + y2) / 2 / image_height, (x2 - x1) / image_width, (y2 - y1) / image_height)
            label_lines.append(" ".join([str(values[0])] + [f"{value:.6f}" for value in values[1:]]))
        (label_dir / f"{source.stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
        image_count += 1
        box_count += len(label_lines)
    return image_count, box_count


def build_yaml(output_dir: Path, class_names: list[str]) -> Path:
    config = {"path": str(output_dir.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "nc": len(class_names), "names": class_names}
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path
