from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from ultralytics import YOLO

REAL_BALL_DIAMETER_CM = 6.7
DEFAULT_FOCAL_LENGTH_PX = 1300.0


def get_project_root(start: Optional[Path] = None) -> Path:
    base = Path.cwd() if start is None else Path(start)
    return base if (base / "tennis_dataset").exists() else base.parent


def get_dataset_dir(project_root: Optional[Path] = None) -> Path:
    root = get_project_root(project_root)
    return root / "tennis_dataset"


def get_data_yaml(project_root: Optional[Path] = None) -> Path:
    return get_dataset_dir(project_root) / "data.yaml"


def get_best_model_path(project_root: Optional[Path] = None) -> Path:
    root = get_project_root(project_root)
    return root / "notebooks" / "runs" / "detect" / "runs_notebook" / "tennis_ball_train" / "weights" / "best.pt"


def load_best_model(project_root: Optional[Path] = None) -> YOLO:
    return YOLO(str(get_best_model_path(project_root)))


def get_largest_ball(results) -> Optional[Tuple[float, float, float, float, float, float]]:
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    largest_box = None
    largest_area = -1.0

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        w = x2 - x1
        h = y2 - y1
        area = w * h

        if area > largest_area:
            largest_area = area
            largest_box = (x1, y1, x2, y2, w, h)

    return largest_box


def estimate_distance_cm(
    width_px: float,
    height_px: float,
    focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
    real_ball_diameter_cm: float = REAL_BALL_DIAMETER_CM,
) -> Optional[float]:
    ball_diameter_px = (width_px + height_px) / 2.0
    if ball_diameter_px <= 0:
        return None
    return (real_ball_diameter_cm * focal_length_px) / ball_diameter_px


def estimate_focal_length_px(
    observed_ball_diameter_px: float,
    known_distance_cm: float,
    real_ball_diameter_cm: float = REAL_BALL_DIAMETER_CM,
) -> Optional[float]:
    if observed_ball_diameter_px <= 0 or real_ball_diameter_cm <= 0:
        return None
    return (known_distance_cm * observed_ball_diameter_px) / real_ball_diameter_cm
