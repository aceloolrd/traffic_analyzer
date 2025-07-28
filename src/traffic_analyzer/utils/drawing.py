# src/traffic_analyzer/utils/drawing.py
import cv2
import numpy as np
from typing import Tuple, List, Dict

def draw_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    text: str = None,
    text_scale: float = 1.5,
    text_thickness: int = 4,
) -> None:
    """
    Рисует прямоугольник и текст на кадре.
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    if text:
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
        bg_x1, bg_y1 = x1, max(0, y1 - ts[1] - 20)
        bg_x2, bg_y2 = x1 + ts[0] + 5, y1 - 10
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        cv2.putText(frame, text, (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, text_thickness)

def draw_zones(frame: np.ndarray, zones: Dict[str, List[Tuple[int,int]]]) -> None:
    """
    Рисует на кадре полигоны зон и их имена.
    """
    if not zones:
        return
    for name, poly in zones.items():
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0,255,255), 2)
        cv2.putText(frame, name, tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

def draw_lines(frame: np.ndarray, lines: Dict[str, Tuple[Tuple[int,int],Tuple[int,int]]]) -> None:
    """
    Рисует на кадре линии и их имена.
    """
    if not lines:
        return
    for name, (p1, p2) in lines.items():
        cv2.line(frame, p1, p2, (0,165,255), 2)
        cv2.putText(frame, name, p1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

def draw_wheels_and_centers(
    frame: np.ndarray,
    wheel_results,
    ox: int, oy: int,
    inlier_mask: np.ndarray = None,
    box_color: Tuple[int, int, int] = (0, 255, 255),
    radius: int = 4
) -> None:
    """
    Рисует боксы вокруг колес и точки центров, инлайеры отмечаются зеленым, а аутлайеры — красным.

    Аргументы:
        frame (np.ndarray): кадр для отрисовки.
        wheel_results: результат детекции колес с атрибутом .boxes.
        ox, oy (int): смещение координат боксов относительно исходного кадра.
        inlier_mask (np.ndarray): булев массив для отметки инлайеров.
        box_color (tuple): цвет бокса колес.
        radius (int): радиус точки центра.
    """
    if not hasattr(wheel_results, 'boxes'):
        return
    for i, box in enumerate(wheel_results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1+ox, y1+oy), (x2+ox, y2+oy), box_color, 1)
        
        cx = int((x1+x2)//2) + ox
        cy = int((y1+y2)//2) + oy
        c = (0,255,0) if (inlier_mask is not None and i < len(inlier_mask) and inlier_mask[i]) else (0,0,255)
        cv2.circle(frame, (cx, cy), radius, c, -1)


def draw_axle_line(frame, centers, ox, oy, ransac_model, color=(255,0,0)):
    if centers is None or len(centers) < 2 or ransac_model is None:
        return
    X = centers[:,0].reshape(-1,1)
    x_vals = np.array([X.min(), X.max()]).reshape(-1,1)
    y_vals = ransac_model.predict(x_vals)
    x_start, y_start = int(x_vals[0][0]) + ox, int(y_vals[0]) + oy
    x_end,   y_end   = int(x_vals[1][0]) + ox, int(y_vals[1]) + oy
    cv2.line(frame, (x_start, y_start), (x_end, y_end), color, 2)
