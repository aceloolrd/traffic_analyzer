# trackers/vehicle_tracker.py

from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict
from datetime import datetime
from _vendor.deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    """
    Обёртка над DeepSort для отслеживания транспортных средств,
    вместе с учётом зон, линий и подготовки данных для CSV.
    """

    # Класс-уровневые хранилища состояния
    TRACK_INFO: Dict[int, Dict[str, Any]] = {}
    LINE_SIDE: Dict[int, Dict[str, int]] = defaultdict(dict)

    def __init__(self, cfg: Dict[str, Any]):
        """
        :param cfg: параметры для DeepSort
        """
        self._tracker = DeepSort(**cfg)

    @classmethod
    def init_track_info(cls, tid: int, frame_idx: int) -> None:
        """
        Создаёт запись о новом треке.
        """
        cls.TRACK_INFO[tid] = {
            "track_id": tid,
            "class_id": None,
            "first_frame": frame_idx,
            "last_frame": frame_idx,
            "start_zone": None,
            "end_zone": None,
            "start_line": None,
            "end_line": None,
            "axles": None,
            "length_frames": 0,
            "enter_time": None,
            "exit_time": None,
        }

    @classmethod
    def finalize_track(cls, tid: int) -> Optional[List[Any]]:
        """
        Завершает трек, возвращает строку для CSV и очищает историю.
        """
        info = cls.TRACK_INFO.pop(tid, None)
        if info is None:
            return None
        row = [
            info["track_id"],
            info["class_id"],
            info["axles"],
            info["start_zone"],
            info["end_zone"],
            info["start_line"],
            info["end_line"],
            info["first_frame"],
            info["last_frame"],
            info["length_frames"],
            info["enter_time"],
            info["exit_time"],
        ]
        cls.LINE_SIDE.pop(tid, None)
        return row

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        frame: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Обновляет треки на основании новых детекций.

        :param detections: список словарей с ключами:
            - 'bbox': [x1, y1, x2, y2]
            - 'confidence': float (опционально)
            - 'class_id': int (опционально)
        :param frame_idx: текущий индекс кадра (для записи в TRACK_INFO)
        :param frame: кадр в BGR (нужен для ReID/embedder)
        :return: подтверждённые треки:
            - 'id':   int
            - 'bbox': [x1, y1, x2, y2]
        """
        # подготовка детекций для DeepSort: (xywh, conf, cls)
        ds_inputs = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            ds_inputs.append((
                [x1, y1, x2 - x1, y2 - y1],
                det.get('confidence', 0.0),
                det.get('class_id', None)
            ))

        tracks = self._tracker.update_tracks(ds_inputs, frame=frame)
        output = []

        for tr in tracks:
            if not tr.is_confirmed():
                continue

            ltrb = tr.to_ltrb()   # [x1,y1,x2,y2] (kalman-predicted)
            if ltrb is None:
                continue

            tid  = tr.track_id
            bbox = [int(v) for v in ltrb]

            # если новый трек — инициализируем
            if tid not in self.TRACK_INFO:
                self.init_track_info(tid, frame_idx)
            info = self.TRACK_INFO[tid]

            # обновляем поля
            info["last_frame"]    = frame_idx
            info["length_frames"] += 1

            output.append({
                "id":   tid,
                "bbox": bbox,
                "det_conf":   det['confidence'], 
                "class_id":   det['class_id']
            })

        return output
