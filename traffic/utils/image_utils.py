# utils/image_utils.py

import numpy as np
from typing import Tuple, Optional

def crop_image(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    padding: int = 0
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Обрезает часть кадра по bbox + padding.
    :return: (crop, ox, oy), где
      - crop: np.ndarray или None, если пусто
      - ox, oy: реальные смещения (x1_p, y1_p) в исходном кадре
    """
    h, w = frame.shape[:2]
    x1_p = max(0, x1 - padding)
    y1_p = max(0, y1 - padding)
    x2_p = min(w, x2 + padding)
    y2_p = min(h, y2 + padding)

    if x2_p <= x1_p or y2_p <= y1_p:
        return None, x1_p, y1_p

    return frame[y1_p:y2_p, x1_p:x2_p], x1_p, y1_p
