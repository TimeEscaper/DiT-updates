import os

from pathlib import Path


def resolve_path(path: str | Path, path_type: str) -> Path:
    """
    Resolves a file system path based on its type, returning an absolute Path.
    Aligns with MIL SberVAE shared directory conventions.

    If the provided path is not absolute, it is joined with the corresponding
    base directory as determined by the path_type parameter and environment variables.

    Args:
        path (str | Path): The file or directory path to be resolved.
        path_type (str): The type of path. Must be one of:
            - "dataset": Uses DATASET_PATH environment variable as base directory.
            - "experiment": Uses EXPERIMENTS_PATH environment variable.
            - "pretrained": Uses PRETRAINED_PATH environment variable.
            - "model": Uses MODELS_PATH environment variable.

    Returns:
        Path: The absolute Path resolved according to the specified path_type.

    Raises:
        ValueError: If the given path_type is not one of the supported options,
                    or if the required environment variable is missing.
    """
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
