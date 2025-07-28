from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from traffic_analyzer.processors.wheel import DEFAULT_AXLES, AXLE_VALUE, LAST_AXLE_FRAME

# Хранилища состояния треков и пересечений
TRACK_INFO: Dict[int, Dict[str, Any]] = {}
LINE_SIDE: Dict[int, Dict[str, int]] = defaultdict(dict)

def init_track_info(tid: int, frame_idx: int) -> None:
    """
    Создаёт запись о новом треке.
    :param tid: идентификатор трека
    :param frame_idx: индекс первого кадра
    """
    TRACK_INFO[tid] = {
        "track_id": tid,
        "class_id": None,
        "first_frame": frame_idx,
        "last_frame": frame_idx,
        "start_zone": None,
        "end_zone": None,
        "start_line": None,
        "end_line": None,
        "axles": DEFAULT_AXLES,
        "length_frames": 0,
        "enter_time": None,
        "exit_time": None,
    }

def finalize_track(tid: int) -> Optional[List[Any]]:
    """
    Завершает трек, возвращает строку для CSV и очищает историю.
    :param tid: идентификатор трека
    :return: список полей (row) или None, если трек не найден
    """
    info = TRACK_INFO.pop(tid, None)
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
    # Очистка связанных storages
    AXLE_VALUE.pop(tid, None)
    LAST_AXLE_FRAME.pop(tid, None)
    LINE_SIDE.pop(tid, None)
    return row
