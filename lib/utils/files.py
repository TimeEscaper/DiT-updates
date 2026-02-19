import os

from pathlib import Path


def resolve_path(path: str | Path, path_type: str) -> Path:
    path = Path(path)
    if not str(path).startswith("/"):
        if path_type == "dataset":
            base_path = os.environ["DATASET_PATH"]
        elif path_type == "experiment":
            base_path = os.environ["EXPERIMENTS_PATH"]
        elif path_type == "pretrained":
            base_path = os.environ["PRETRAINED_PATH"]
        elif path_type == "model":
            base_path = os.environ["MODELS_PATH"]
        else:
            raise ValueError(f"Unknown path type: {path_type}")
        path = base_path / path
    return path

