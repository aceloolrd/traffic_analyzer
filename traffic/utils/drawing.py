# utils/drawing.py

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class FrameDrawer:
    """
    Универсальный рисователь боксов на RGB-кадре.
    Поддерживает смещение offset и поля 'label'/'confidence'.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.vehicle_color   = tuple(cfg.get('vehicle_box_color', [0,255,0]))
        self.wheel_color     = tuple(cfg.get('wheel_box_color',   [255,0,0]))
        self.box_label_color      = tuple(cfg.get('box_label_color',        [255,255,255]))
        self.text_color      = tuple(cfg.get('text_color',        [0,0,0]))
        self.box_thickness   = cfg.get('box_thickness', 1)
        self.text_scale      = cfg.get('text_scale',    0.5)
        self.text_thickness  = cfg.get('text_thickness',1)
        self.font            = cfg.get('font', cv2.FONT_HERSHEY_SIMPLEX)
        
         # ——— Настройки рисования центров колёс и оси ———
        self.center_inlier_color   = tuple(cfg.get('center_inlier_color', [0,255,0]))
        self.center_outlier_color  = tuple(cfg.get('center_outlier_color', [0,0,255]))
        self.center_radius   = cfg.get('center_radius', 5)
        self.axle_color      = tuple(cfg.get('axle_color', [255,0,0]))
        self.axle_thickness  = cfg.get('axle_thickness', 2)

    def draw_boxes(self, frame: np.ndarray, anns: List[Dict[str, Any]]) -> None:
        """
        anns элемент:

        {
            'id':         int,        # optionalid
            'bbox':       (x1, y1, x2, y2)
            'type':       'vehicle'|'wheel'
            'cls':        str,        # optional
            'det_conf':   float,      # optional
            'cls_conf':   float,      # optional
            'offset':     (dx,dy)     # optional, default (0,0)
        }
        """
        for ann in anns:
            typ            = ann.get('type')
            x1, y1, x2, y2 = ann.get('bbox')
            det_conf           = ann.get('det_conf', None)
            dx, dy         = ann.get('offset', (0, 0))
            
            if typ == 'vehicle':
                color_rgb = self.vehicle_color
                color_bgr = color_rgb[::-1]
                id             = ann.get('id')
                lbl            = ann.get('cls', '')
                cls_conf       = ann.get('cls_conf', None)

            
            elif typ == 'wheel':
                color_rgb = self.wheel_color
                color_bgr = color_rgb[::-1]
                id             = None
                lbl            = None
                cls_conf       = None
                

            # рисуем рамку с учётом offset
            sx, sy = x1+dx, y1+dy
            ex, ey = x2+dx, y2+dy
            cv2.rectangle(frame, (sx, sy), (ex, ey), color_bgr, self.box_thickness)

            # формируем текст
            parts = []
            
            # 1) ID трека
            if id is not None:
                parts.append(f"id:{id}")
            # 2) уверенность детектора
            if det_conf is not None:
                parts.append(f"dc: {float(det_conf):.2f}")
            # 3) название класса
            if lbl is not None:
                parts.append(f"cls: {lbl}")
            # 4) уверенность классификатора
            if cls_conf is not None:
                parts.append(f"cc: {float(cls_conf):.2f}")
            if not parts:
                continue       
            
            # 5) объединяем части текста   
            text = " ".join(parts)

            # вычисляем размер текста
            (tw, th), _ = cv2.getTextSize(text, self.font, self.text_scale, self.text_thickness)

            # фон под текст
            bg_x1 = sx
            bg_y2 = sy
            bg_y1 = max(0, bg_y2 - th - 2*self.text_thickness)
            bg_x2 = bg_x1 + tw + self.text_thickness

            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), self.box_label_color, -1)

            # рисуем текст
            cv2.putText(
                frame, text,
                (bg_x1, bg_y2 - self.text_thickness),
                self.font,
                self.text_scale,
                self.text_color[::-1],
                self.text_thickness,
                cv2.LINE_AA
            )

    def draw_zones(self, frame: np.ndarray, zones: Dict[str, List[Tuple[int,int]]]) -> None:
        """
        Рисует зоны (полигоны) и их имена.
        zones: dict zone_name -> list of (x,y) vertices
        """
        for name, pts in zones.items():
            if not pts: continue
            # рисуем замкнутый контур
            arr = np.array(pts, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(frame, [arr], isClosed=True, color=self.zone_color[::-1],
                          thickness=self.region_thickness)
            # подпись в первой вершине
            x0,y0 = pts[0]
            cv2.putText(
                frame, name,
                (x0, y0 - 5),
                self.font, self.region_font_scale,
                self.zone_color[::-1],
                self.text_thickness,
                cv2.LINE_AA
            )

    def draw_lines(self, frame: np.ndarray, lines: Dict[str, Tuple[Tuple[int,int],Tuple[int,int]]]) -> None:
        """
        Рисует линии и их имена.
        lines: dict line_name -> ((x1,y1),(x2,y2))
        """
        for name, (p1,p2) in lines.items():
            cv2.line(frame, p1, p2, self.line_color[::-1], self.region_thickness)
            # подпись в середине
            mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
            cv2.putText(
                frame, name,
                (mx+5, my-5),
                self.font, self.region_font_scale,
                self.line_color[::-1],
                self.text_thickness,
                cv2.LINE_AA
            )
    
    def draw_wheel_centers_and_axle(
        self,
        frame: np.ndarray,
        centers: np.ndarray,
        offset: Tuple[int,int] = (0,0),
        inlier_mask: Optional[np.ndarray] = None,
        ransac_model: Optional[Any] = None
    ) -> None:
        """
        Рисует кружки в точках центров и осевую линию по RANSAC-модели.
        """
        ox, oy = offset

        # 1) рисуем кружки центров
        for i, (cx, cy) in enumerate(centers):
            x, y = int(cx)+ox, int(cy)+oy
            if inlier_mask is not None and i < len(inlier_mask): # проверяем, что индекс в пределах маски
                c = self.center_inlier_color if inlier_mask[i] else self.center_outlier_color
            else:
                c = self.center_inlier_color # если маски нет, то считаем все центры инлайнерами
            cv2.circle(frame, (x,y), self.center_radius, c, -1)

        # 2) рисуем линию оси
        if ransac_model is None or len(centers) < 2:
            return

        X = centers[:,0].reshape(-1,1)
        xv = np.array([X.min(), X.max()]).reshape(-1,1)
        yv = ransac_model.predict(xv)

        p1 = (int(xv[0][0])+ox, int(yv[0])+oy)
        p2 = (int(xv[1][0])+ox, int(yv[1])+oy)
        cv2.line(frame, p1, p2, self.axle_color, self.axle_thickness)