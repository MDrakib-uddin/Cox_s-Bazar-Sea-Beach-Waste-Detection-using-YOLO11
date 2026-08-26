from pathlib import Path

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def find_dataset(root: Path) -> Path:
    candidates = [root, *root.rglob("*")]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "train" / "_annotations.coco.json").exists():
            return candidate
    raise FileNotFoundError(
        f"COCO dataset not found under {root}. Expected train/_annotations.coco.json."
    )


def get_paths(data_root: Path | None = None, output_dir: Path | None = None, runs_dir: Path | None = None):
    kaggle = Path("/kaggle").exists()
    requested_root = data_root or (Path("/kaggle/input/datasets") if kaggle else Path.cwd() / "dataset")
    dataset_dir = find_dataset(requested_root)
    yolo_dir = output_dir or (Path("/kaggle/working/dataset_yolo") if kaggle else dataset_dir.parent / "dataset_yolo")
    training_dir = runs_dir or (Path("/kaggle/working/runs") if kaggle else dataset_dir.parent / "runs")
    yolo_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir, yolo_dir, training_dir
