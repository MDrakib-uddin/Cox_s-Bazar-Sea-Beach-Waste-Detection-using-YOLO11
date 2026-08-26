from __future__ import annotations

import argparse
from pathlib import Path

from coco_converter import build_yaml, convert_split, load_classes
from config import SPLITS, get_paths
from model_tasks import evaluate_model, predict_one, save_best_model, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on the COCO plastic-waste dataset.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    dataset_dir, yolo_dir, runs_dir = get_paths(args.data_root, args.output_dir, args.runs_dir)
    category_map, class_names = load_classes(dataset_dir)
    for output_split, source_split in SPLITS.items():
        image_count, box_count = convert_split(dataset_dir / source_split, yolo_dir, output_split, category_map)
        print(f"{output_split}: {image_count} images, {box_count} boxes")

    yaml_path = build_yaml(yolo_dir, class_names)
    print(f"Dataset: {dataset_dir}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Config: {yaml_path}")
    print(f"Runs: {runs_dir}")

    device, best_weights = train_model(yaml_path, runs_dir, args.weights, args.epochs, args.batch, args.imgsz)
    best_model = evaluate_model(best_weights, yaml_path, args.imgsz, device)
    save_best_model(best_weights, runs_dir)
    if args.predict:
        predict_one(best_model, yolo_dir, device)


if __name__ == "__main__":
    main()
