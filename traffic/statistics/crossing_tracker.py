# statistics/crossing_tracker.py

import numpy as np
from typing import Dict, Tuple, Any, List
from datetime import datetime


class CrossingTracker:
    """
    Статистика пересечений зон (полигоны) и линий для треков.

    При инициализации принимает:
      - zones: Dict[name, List[(x,y),...]]
      - lines: Dict[name, ((x1,y1),(x2,y2)), ...]
    """

    def __init__(
        self,
        zones: Dict[str, List[Tuple[int,int]]],
        lines: Dict[str, Tuple[Tuple[int,int], Tuple[int,int]]],
    ):
        self.zones = zones or {}
        self.lines = lines or {}
        # для каждого track_id храним последний знак стороны для каждой линии
        self._line_side: Dict[int, Dict[str, int]] = {}

    def update(
        self,
        track_id: int,
        cx: int,
        cy: int,
        info: Dict[str, Any],
        frame_idx: int
    ) -> None:
        """
        Вызывается каждый кадр для каждого трека,
        обновляет info["start_zone"/"end_zone"] и info["start_line"/"end_line"].
        """
        # 1) зоны
        zone = self._get_zone_for_point(cx, cy)
        if info["start_zone"] is None and zone:
            info["start_zone"]   = zone
            info["enter_time"]   = datetime.now().isoformat()
        info["end_zone"] = zone
        info["exit_time"] = datetime.now().isoformat()

        # 2) линии
        if track_id not in self._line_side:
            self._line_side[track_id] = {}
        for name, (p1, p2) in self.lines.items():
            prev = self._line_side[track_id].get(name, 0)
            cur  = self._side_of_line(cx, cy, p1, p2)
            self._line_side[track_id][name] = cur

            if self._crossed(prev, cur) and self._point_between_segment(cx, cy, p1, p2):
                if info["start_line"] is None:
                    info["start_line"] = name
                    info["enter_time"] = datetime.now().isoformat()
                info["end_line"] = name
                info["exit_time"] = datetime.now().isoformat()

    # --- Вспомогательные "старые" геометрические функции: ---
    def _point_in_poly(self, x: int, y: int, poly: List[Tuple[int,int]]) -> bool:
        inside = False
        n = len(poly)
        px, py = x, y
        x1, y1 = poly[0]
        for i in range(1, n + 1):
            x2, y2 = poly[i % n]
            if min(y1, y2) < py <= max(y1, y2) and px <= max(x1, x2):
                if y1 != y2:
                    xinters = (py - y1) * (x2 - x1) / (y2 - y1 + 1e-9) + x1
                if x1 == x2 or px <= xinters:
                    inside = not inside
            x1, y1 = x2, y2
        return inside

    def _get_zone_for_point(self, x: int, y: int) -> Any:
        for name, poly in self.zones.items():
            if self._point_in_poly(x, y, poly):
                return name
        return None

    def _side_of_line(
        self,
        px: int, py: int,
        p1: Tuple[int,int],
        p2: Tuple[int,int]
    ) -> int:
        return int(np.sign(
            (px - p1[0])*(p2[1] - p1[1]) - (py - p1[1])*(p2[0] - p1[0])
        ))

    def _point_between_segment(
        self,
        px: int, py: int,
        p1: Tuple[int,int],
        p2: Tuple[int,int],
        margin: int = 5
    ) -> bool:
        xmin, xmax = sorted([p1[0], p2[0]])
        ymin, ymax = sorted([p1[1], p2[1]])
        return (xmin - margin <= px <= xmax + margin) and (ymin - margin <= py <= ymax + margin)

    def _crossed(self, prev: int, cur: int) -> bool:
        return prev != 0 and cur != 0 and prev != cur
