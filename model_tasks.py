from __future__ import annotations

import random
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def train_model(yaml_path: Path, runs_dir: Path, weights: str, epochs: int, batch: int, imgsz: int):
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(weights)
    model.train(data=str(yaml_path), epochs=epochs, imgsz=imgsz, batch=batch if device == 0 else min(batch, 4), device=device, workers=0, patience=20, project=str(runs_dir), name="plastic_waste_yolo11n", exist_ok=True, pretrained=True, plots=True)
    return device, runs_dir / "plastic_waste_yolo11n" / "weights" / "best.pt"


def evaluate_model(best_weights: Path, yaml_path: Path, imgsz: int, device):
    if not best_weights.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_weights}")
    model = YOLO(str(best_weights))
    metrics = model.val(data=str(yaml_path), split="test", imgsz=imgsz, device=device)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    return model


def save_best_model(best_weights: Path, runs_dir: Path) -> Path:
    model_file = runs_dir / "plastic_waste_yolo11n_best.pt"
    shutil.copy2(best_weights, model_file)
    print(f"Model saved: {model_file}")
    return model_file


def predict_one(model, yolo_dir: Path, device) -> None:
    test_images = [path for path in (yolo_dir / "images" / "test").glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not test_images:
        raise FileNotFoundError("No test images found for prediction.")
    prediction = model.predict(source=str(random.choice(test_images)), conf=0.25, save=True, device=device)
    print(f"Prediction saved: {prediction[0].save_dir}")
