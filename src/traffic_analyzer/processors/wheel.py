import numpy as np
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor
from collections import defaultdict, deque, Counter
from typing import Tuple, List, Dict

from traffic_analyzer.utils.drawing import draw_wheels_and_centers, draw_axle_line

# Кэш YOLO‑детекторов колёс по model_path
_wheel_detectors: Dict[str, YOLO] = {}

AXLE_HISTORY = defaultdict(lambda: deque(maxlen=5))
AXLE_VALUE = {}
LAST_AXLE_FRAME = {}
DEFAULT_AXLES = 2

def update_axles(track_id: int, count: int):
    if not count or count < DEFAULT_AXLES:
        return
    AXLE_HISTORY[track_id].append(count)
    AXLE_VALUE[track_id] = Counter(AXLE_HISTORY[track_id]).most_common(1)[0][0]

def ensure_default_axles(track_id: int):
    if track_id not in AXLE_VALUE:
        AXLE_VALUE[track_id] = DEFAULT_AXLES

def crop_vehicle(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    padding: int = 0
) -> Tuple[np.ndarray, int, int]:
    h, w, _ = frame.shape
    x1_p = max(0, x1 - padding)
    y1_p = max(0, y1 - padding)
    x2_p = min(w, x2 + padding)
    y2_p = min(h, y2 + padding)
    if x2_p <= x1_p or y2_p <= y1_p:
        return None, x1_p, y1_p
    return frame[y1_p:y2_p, x1_p:x2_p], x1_p, y1_p

def detect_wheels(
    crop: np.ndarray,
    model_path: str,
    conf: float,
    iou: float,
    half: bool = True,
    device: str = '0',
    max_det: int = 20,
    verbose: bool = False,
) -> object:
    """
    Выполняет детекцию колёс на обрезанном crop’е.
    Параметры (model_path, conf, iou и пр.) передаются извне.
    """
    class Dummy:
        boxes: List = []
    if crop is None or crop.size == 0:
        return Dummy()

    # лениво инициализируем YOLO-детектор для данного пути
    if model_path not in _wheel_detectors:
        _wheel_detectors[model_path] = YOLO(model_path)
    detector = _wheel_detectors[model_path]

    return detector(
        crop,
        conf=conf,
        iou=iou,
        half=half,
        device=device,
        max_det=max_det,
        verbose=verbose
    )[0]

def fit_line_with_ransac(
    points: np.ndarray,
    residual_threshold: float = 5,
    max_trials: int = 100
) -> Tuple[np.ndarray, object]:
    n = len(points)
    if n < 2:
        return np.zeros(n, dtype=bool), None
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    if np.var(X) < 1e-6 or np.var(y) < 1e-6:
        return np.ones(n, dtype=bool), None
    try:
        ransac = RANSACRegressor(
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            min_samples=2,
            random_state=42
        )
        ransac.fit(X, y)
        return ransac.inlier_mask_, ransac
    except ValueError:
        return np.ones(n, dtype=bool), None

def process_wheels_on_vehicle(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    model_path: str,
    conf: float,
    iou: float,
    padding: int = 0,
    residual_threshold: float = 5,
    half: bool = True,
    device: str = '0',
    max_det: int = 20,
    verbose: bool = False,
) -> Tuple[str, int]:
    """
    Рисует колёса и ось, считает число осей.
    Все параметры детектора и RANSAC передаются извне.
    """
    x1, y1, x2, y2 = box
    crop, ox, oy = crop_vehicle(frame, x1, y1, x2, y2, padding)
    wheel_results = detect_wheels(
        crop, model_path, conf, iou, half, device, max_det, verbose
    )

    centers = []
    for wb in getattr(wheel_results, "boxes", []):
        x1w, y1w, x2w, y2w = map(int, wb.xyxy[0])
        centers.append([(x1w + x2w) / 2, (y1w + y2w) / 2])
    centers = np.array(centers)

    if len(centers) == 0:
        draw_wheels_and_centers(frame, wheel_results, ox, oy)
        return " ax:2", DEFAULT_AXLES

    if len(centers) >= 2:
        mask, model = fit_line_with_ransac(centers, residual_threshold)
        draw_wheels_and_centers(frame, wheel_results, ox, oy, inlier_mask=mask)
        draw_axle_line(frame, centers, ox, oy, model)
        axes = max(int(mask.sum()), DEFAULT_AXLES)
        return f" ax:{axes}", axes

    draw_wheels_and_centers(frame, wheel_results, ox, oy)
    return " ax:2", DEFAULT_AXLES
