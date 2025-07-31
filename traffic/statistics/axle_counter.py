# statistics/axle_counter.py

import numpy as np
from sklearn.linear_model import RANSACRegressor
from collections import defaultdict, deque, Counter
from typing import Tuple, List, Dict, Any, Optional

from detectors.detector import Detector
from utils.image_utils import crop_image    # <- обновлённый импорт


class AxleCounter:
    """
    Считает оси «стабильно» и отдаёт всё, что нужно для отрисовки.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.wheel_detector   = Detector(cfg['wheels_detector'])
        self.ransac_residual  = cfg.get('ransac_residual_threshold', 5)
        self.ransac_max_trials= cfg.get('ransac_max_trials', 100)
        self.history_len      = cfg.get('history_len', 5)
        self.default_axles    = cfg.get('default_axles', 2)

        self._history = defaultdict(lambda: deque(maxlen=self.history_len))
        self._values  = {}  # track_id -> int

    def _fit_ransac(self, centers: np.ndarray) -> Tuple[np.ndarray, Optional[RANSACRegressor]]:
        n = len(centers)
        if n < 2:
            return np.zeros(n, dtype=bool), None
        X = centers[:,0].reshape(-1,1)
        y = centers[:,1]
        try:
            r = RANSACRegressor(
                residual_threshold=self.ransac_residual,
                max_trials=self.ransac_max_trials,
                min_samples=2,
                random_state=42
            )
            r.fit(X, y)
            return r.inlier_mask_, r
        except ValueError:
            return np.ones(n, dtype=bool), None

    def process(
        self,
        frame: np.ndarray,
        bbox: Tuple[int,int,int,int],
        track_id: int,
        frame_idx: int,
        padding: int = 0
    ) -> Tuple[
          int,                        # final axle count
          List[Dict[str,Any]],        # wheel detections
          np.ndarray,                 # centers
          np.ndarray,                 # inlier mask
          Optional[RANSACRegressor],  # model
          int, int                    # ox, oy
    ]:
        x1, y1, x2, y2 = bbox
        crop, ox, oy = crop_image(frame, x1, y1, x2, y2, padding)
        if crop is None or crop.size == 0:
            return (
                self.default_axles, [], 
                np.zeros((0,2)), np.zeros(0, dtype=bool),
                None, ox, oy
            )

        # 1) detect wheels
        wheel_dets = self.wheel_detector.detect(crop)

        # 2) compute centers
        centers = np.array([
            [(b[0]+b[2]) / 2, (b[1]+b[3]) / 2]
            for d in wheel_dets for b in [d['bbox']]
        ])

        # 3) RANSAC
        if len(centers) >= 2:
            mask, model = self._fit_ransac(centers)
            raw_axes = int(mask.sum())
        else:
            mask, model = np.zeros(len(centers), bool), None
            raw_axes = self.default_axles

        # 4) stabilization
        if raw_axes >= self.default_axles:
            self._history[track_id].append(raw_axes)
            m = Counter(self._history[track_id]).most_common(1)[0][0]
            self._values[track_id] = m
        else:
            self._values.setdefault(track_id, self.default_axles)

        return (
            self._values[track_id],
            wheel_dets,
            centers,
            mask,
            model,
            ox, oy
        )
