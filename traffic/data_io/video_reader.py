# data_io/video_reader.py

import os
from typing import Generator, Optional, Dict, Any
from PIL import Image
import cv2

from data_io.sff_reader import SFFReader

class VideoReader:
    """
    Универсальный ридер кадров: .sff/.dat или видеофайл.
    Поддерживает пропуск кадров (frame_skip).
    Инициализируется одним словарём config['input'].
    Возвращает PIL.Image кадры.
    """
    def __init__(self, cfg: Dict[str, Any]):
        """
        :param cfg: словарь с ключами:
            - video_path: str        # путь к .sff или видеофайлу
            - frame_skip: int        # брать каждый N-й кадр
            - picket_thr: int        # порог для SFFReader.get_frames
            - reverse: bool          # инвертировать порядок в SFF
        """
        path = cfg['video_path']
        self.frame_skip   = max(1, cfg.get('frame_skip', 1))
        self._ext         = os.path.splitext(path)[1].lower()
        self._is_sff      = self._ext == ".sff"
        self._path        = path

        if self._is_sff:
            self._sff         = SFFReader(path)
            self._picket_thr  = cfg.get('picket_thr', 2)
            self._reverse     = cfg.get('reverse', False)
        else:
            self._cap: Optional[cv2.VideoCapture] = None

    def __iter__(self) -> Generator[Image.Image, None, None]:
        if self._is_sff:
            return self._iter_sff()
        else:
            return self._iter_video()

    def _iter_sff(self) -> Generator[Image.Image, None, None]:
        """
        Через SFFReader.get_frames()
        """
        for idx, img_pil, _, _ in self._sff.get_frames(
            reverse=self._reverse,
            picket_thr=self._picket_thr
        ):
            if idx % self.frame_skip != 0:
                continue
            yield img_pil

    def _iter_video(self) -> Generator[Image.Image, None, None]:
        """
        Через OpenCV VideoCapture.
        """
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self._path}")

        idx = 0
        while True:
            ret, frame_bgr = self._cap.read()
            if not ret:
                break

            if idx % self.frame_skip == 0:
                # BGR → RGB → PIL.Image
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield Image.fromarray(frame_rgb)

            idx += 1

        self._cap.release()
