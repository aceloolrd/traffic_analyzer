# pipelines/traffic_pipeline.py

import os
import cv2
import numpy as np
from typing import Dict, Any

from pipelines.base import BaseRecognitionPipeline

from detectors.detector import Detector
from classifiers.vehicle_classifier import VehicleClassifier
from trackers.vehicle_tracker import VehicleTracker

from statistics.crossing_tracker import CrossingTracker
from statistics.axle_counter import AxleCounter

from data_io.video_reader import VideoReader
from data_io.csv_writer import CSVWriter

from utils.drawing import FrameDrawer
from utils.geometry import should_calc_axle
from utils.image_utils import crop_image


class TrafficPipeline(BaseRecognitionPipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._init_components()

    def _init_components(self):
        cfg = self.config

        # 1) Reader (.sff или видео)
        self.reader = VideoReader(cfg['input'])

        # 2) Модели
        self.vehicle_detector   = Detector(cfg['vehicle_detector'])
        self.vehicle_classifier = VehicleClassifier(cfg['classifier'])
        self.vehicle_tracker    = VehicleTracker(cfg['tracker'])

        # 3) Пересечения зон/линий
        self.crossing_tracker = CrossingTracker(
            zones=cfg['pipeline'].get('zones', {}),
            lines=cfg['pipeline'].get('lines', {})
        )

        # 4) Счетчик осей
        self.axle_counter = AxleCounter(cfg['wheels_counter'])

        # Настройки пайплайна
        p = cfg['pipeline']
        self.use_tracking = p.get('use_tracking', True) # использовать трекинг
        self.use_axle     = p.get('use_axle', True) # использовать подсчет осей
        self.log_every    = p.get('log_every', 10) # логировать каждые N кадров

        # Вывод
        o = cfg['output']
        os.makedirs(os.path.dirname(o['csv_path']), exist_ok=True)
        self.writer       = CSVWriter(o['csv_path'])
        self.drawer       = FrameDrawer(o)
        self.draw_regions = o.get('draw_regions', True) # отрисовывать регионы зон/линий

        # Видео на выход (опционально)леле
        self.save_video = bool(o.get('out_video_path'))
        self.out_path   = o.get('out_video_path', "")
        self.out_fps    = o.get('out_fps', 25)

    def run(self):
        writer = None
        frame_idx = 0

        for pil_frame in self.reader:

            # Подготовка кадра
            frame_rgb = np.array(pil_frame)  # PIL → RGB numpy
            h, w = frame_rgb.shape[:2]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Инициализация VideoWriter
            if self.save_video and writer is None:
                writer = cv2.VideoWriter(
                    self.out_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.out_fps,
                    (w, h)
                )

            # 1) Детект авто (на RGB)
            dets = self.vehicle_detector.detect(frame_rgb)
            '''
            {
                "bbox":       [x1, y1, x2, y2],
                "confidence": conf,
                "class_id":   cls
            }
            '''
            
            # 2) Трекинг
            if self.use_tracking:
                tracks = self.vehicle_tracker.update(
                    detections=dets,
                    frame=frame_bgr,
                    frame_idx=frame_idx
                )
                '''
                tracks формат:
                [{
                    "id":   tid,
                    "bbox": bbox,
                    "det_conf":   det['confidence'], 
                    "class_id":   det['class_id']
                }]
                '''
                alive_ids = {trk['id'] for trk in tracks} # id живых треков
            else: 
                tracks = [{
                    'id':           i, # если трекинг не используется, то id просто по индексу
                    'bbox':         d['bbox'],
                    'det_conf':     d['det_conf'],
                    'class_id':     d['class_id']
                } for i, d in enumerate(dets)]
                alive_ids = set(trk['id'] for trk in tracks)

            veh_anns, wheel_anns = [], []

            # 3) Обработка каждого трека
            for tr in tracks:
                tid = tr['id']
                x1, y1, x2, y2 = tr['bbox']

                # 3.1) Классификация (на RGB-кропе)
                crop_rgb, _, _ = crop_image(frame_rgb, x1, y1, x2, y2)
                cls = None
                if crop_rgb is not None:
                    cls_id, cls, cls_conf = self.vehicle_classifier.classify(crop_rgb)

                # 3.2) Обновляем статистику по зонам/линиям
                # центр бокса объекта
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                if tid not in self.vehicle_tracker.TRACK_INFO: # если трек новый
                    self.vehicle_tracker.init_track_info(tid, frame_idx) # инициализируем его
                '''
                TRACK_INFO - это словарь, где ключ - id трека, а значение - словарь с информацией о треке:
                {
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
                '''
                info = self.vehicle_tracker.TRACK_INFO[tid] # получаем информацию о треке
                self.crossing_tracker.update(tid, cx, cy, info, frame_idx) # обновляем статистику

                # 3.3) Подсчет осей
                if self.use_axle and should_calc_axle(
                    x2, y2, w, h,
                    True,
                    self.config['pipeline'].get('hz_start'),
                    self.config['pipeline'].get('hz_stop'),
                    self.config['pipeline'].get('vz_start'),
                    self.config['pipeline'].get('vz_stop'),
                ):
                    axle_count, wheel_dets, centers, mask, model, ox, oy = \
                        self.axle_counter.process(
                            frame_bgr,
                            (x1, y1, x2, y2),
                            tid, frame_idx,
                            padding=self.config['wheels_counter'].get('padding', 0)
                        )
                else:
                    axle_count = self.axle_counter.default_axles # если оси не считаем, то возвращаем дефолтное значение - 2
                    wheel_dets, centers, mask, model, ox, oy = [], np.zeros((0,2)), np.zeros(0), None, 0, 0 # пустые значения

                # Обновляем TRACK_INFO
                info.update({
                    "class_id":     cls,
                    "axles":        axle_count,
                    "last_frame":   frame_idx,
                    "length_frames": info.get("length_frames", 0) + 1
                })

                # 3.4) Готовим аннотации
                veh_anns.append({
                    'id':         tid,
                    'bbox':       (x1, y1, x2, y2),
                    'type':       'vehicle',
                    'cls':        cls,
                    'det_conf':   tr['det_conf'],    # уверенность ДЕТЕКТОРА
                    'cls_conf':   cls_conf,    # уверенность КЛАССИФИКАТОРА
                    'offset':     (0, 0)
                })
                for wd in wheel_dets:
                    bx1, by1, bx2, by2 = wd['bbox']
                    wheel_anns.append({
                        'bbox':       (bx1, by1, bx2, by2),
                        'type':       'wheel',
                        'det_conf':   wd.get('confidence'),    # уверенность ДЕТЕКТОРА
                        'offset':     (ox, oy)
                    })

            # 4) Отрисовка (в порядке: боксы, центры+оси, регионы)
            self.drawer.draw_boxes(frame_bgr, veh_anns + wheel_anns)
            
            if self.use_axle: # если считаем оси, то рисуем центры и оси
                for tr in tracks:
                    tid = tr['id']
                    x1, y1, x2, y2 = tr['bbox']
                    _, _, centers, mask, model, _, _ = \
                        self.axle_counter.process( # считаем оси
                            frame_bgr,
                            (x1, y1, x2, y2),
                            tid, frame_idx,
                            padding=self.config['wheels_counter'].get('padding', 0)
                        )
                    self.drawer.draw_wheel_centers_and_axle( # рисуем центры и оси
                        frame_bgr,
                        centers,
                        offset=(x1, y1),
                        inlier_mask=mask,
                        ransac_model=model
                    )
            if self.draw_regions: # если нужно, то рисуем регионы зон/линий
                self.drawer.draw_zones(frame_bgr, self.crossing_tracker.zones)
                self.drawer.draw_lines(frame_bgr, self.crossing_tracker.lines)

            # 5) Финализация исчезнувших треков
            for tid in list(self.vehicle_tracker.TRACK_INFO):
                if tid not in alive_ids:
                    row = self.vehicle_tracker.finalize_track(tid)
                    if row:
                        self.writer.write_row(row) # записываем в CSV

            # 6) Показ и сохранение видео
            cv2.imshow("Traffic", frame_bgr)
            if self.save_video:
                writer.write(frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1
            if frame_idx % self.log_every == 0:
                print(f"[INFO] processed {frame_idx} frames")

        # 7) Завершаем всё
        for tid in list(self.vehicle_tracker.TRACK_INFO):
            row = self.vehicle_tracker.finalize_track(tid)
            if row:
                self.writer.write_row(row)

        if writer:
            writer.release()
        self.writer.close()
        cv2.destroyAllWindows()
