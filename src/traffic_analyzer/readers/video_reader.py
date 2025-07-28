import cv2
import numpy as np
from typing import Generator, Tuple, Dict

def frames_from_video(video_path: str) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """
    Генератор кадров из видеофайла через OpenCV.
    :param video_path: путь к видео
    :yield: (frame_bgr, info_dict)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, {"frame_num": idx, "fps": fps, "size": (w, h)}
        idx += 1
    cap.release()
