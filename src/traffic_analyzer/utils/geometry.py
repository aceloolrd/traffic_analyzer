import numpy as np
from typing import Tuple, List

# Функции работы с полигонами и линиями
def should_calc_axle(
    x2: int, y2: int, frame_w: int, frame_h: int,
    axle_for_all: bool,
    hz_start: int, hz_stop: int,
    vz_start: int, vz_stop: int
) -> bool:
    """
    Определяет, нужно ли выполнять расчёт осей по заданным зонам.

    Аргументы:
        x2, y2 (int): координаты правого нижнего угла бокса автомобиля.
        frame_w, frame_h (int): ширина и высота кадра.
        axle_for_all (bool): флаг обхода зонной фильтрации.
        hz_start, hz_stop (int): границы горизонтальной зоны.
        vz_start, vz_stop (int): границы вертикальной зоны.

    Возвращает:
        bool: True, если расчёт осей разрешён.
    """
    if axle_for_all:
        return True
    ok_h = True
    ok_v = True
    if hz_start is not None and hz_stop is not None:
        d_right = frame_w - x2
        ok_h = (d_right <= hz_start) and (d_right >= hz_stop)
    if vz_start is not None and vz_stop is not None:
        d_bottom = frame_h - y2
        ok_v = (d_bottom <= vz_start) and (d_bottom >= vz_stop)
    return ok_h and ok_v

def point_in_poly(x: int, y: int, poly: List[Tuple[int, int]]) -> bool:
    """
    Проверяет, находится ли точка внутри многоугольника.

    Аргументы:
        x, y (int): координаты точки.
        poly (List[Tuple[int,int]]): список вершин многоугольника.

    Возвращает:
        bool: True, если точка внутри.
    """
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

def get_zone_for_point(x: int, y: int, zones: dict) -> str:
    """
    Определяет имя зоны, в которую попадает точка.

    Аргументы:
        x, y (int): координаты точки.
        zones (dict): словарь зон {имя: poly}.

    Возвращает:
        str: имя зоны или None.
    """
    if not zones:
        return None
    for name, poly in zones.items():
        if point_in_poly(x, y, poly):
            return name
    return None

def side_of_line(px: int, py: int, p1: Tuple[int,int], p2: Tuple[int,int]) -> int:
    """
    Вычисляет сторону точки относительно прямой.

    Аргументы:
        px, py (int): координаты точки.
        p1, p2 (Tuple[int,int]): точки, задающие прямую.

    Возвращает:
        int: знак положения (1, -1 или 0).
    """
    return np.sign((px - p1[0]) * (p2[1] - p1[1]) - (py - p1[1]) * (p2[0] - p1[0]))

def point_between_segment(
    px: int, py: int, p1: Tuple[int,int], p2: Tuple[int,int], margin: int = 5
) -> bool:
    """
    Проверяет, лежит ли точка в пределах отрезка с запасом.

    Аргументы:
        px, py (int): координаты точки.
        p1, p2 (Tuple[int,int]): концы отрезка.
        margin (int): дополнительный допуск.

    Возвращает:
        bool: True, если точка в пределах.
    """
    xmin, xmax = sorted([p1[0], p2[0]])
    ymin, ymax = sorted([p1[1], p2[1]])
    return (xmin - margin <= px <= xmax + margin) and (ymin - margin <= py <= ymax + margin)

def crossed(prev: int, cur: int) -> bool:
    """
    Определяет, пересекла ли точка линию (изменился знак).

    Аргументы:
        prev, cur (int): прежнее и текущее значение стороны.

    Возвращает:
        bool: True при пересечении.
    """
    return prev != 0 and cur != 0 and prev != cur