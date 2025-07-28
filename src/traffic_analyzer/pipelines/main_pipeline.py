import cv2
import random
from datetime import datetime
from traffic_analyzer._vendor.deep_sort_realtime.deepsort_tracker import DeepSort


from traffic_analyzer.config import CONFIG

from traffic_analyzer.processors.car import detect_cars, classify_car
from traffic_analyzer.processors.wheel import (
    process_wheels_on_vehicle,
    update_axles,
    ensure_default_axles,
    AXLE_VALUE, LAST_AXLE_FRAME, DEFAULT_AXLES
)
from traffic_analyzer.utils.drawing import draw_box, draw_zones, draw_lines
from traffic_analyzer.utils.geometry import (
    get_zone_for_point, side_of_line,
    point_between_segment, crossed, should_calc_axle
)

from traffic_analyzer.readers.sff_reader import SFFReader, frames_from_sff
from traffic_analyzer.readers.video_reader import frames_from_video

from traffic_analyzer.utils.logging import dbg
from traffic_analyzer.utils.tracing import init_track_info, finalize_track, TRACK_INFO, LINE_SIDE
from traffic_analyzer.utils.csv_writer import write_csv, CSV_HEADER
from traffic_analyzer.processors.wheel import crop_vehicle

def _run_core(
    frame_gen,
    config: dict = CONFIG,
    use_tracking: bool = True,
    show_track_id: bool = True,
    use_axle: bool = False,
    debug: bool = False,
    log_every: int = 10,
    save_out: bool = False,
    out_path: str = "output.mp4",
    out_fps: int = None,
    zones: dict = None,
    lines: dict = None,
    draw_regions: bool = True,
    csv_path: str = "objects.csv",
):
    dbg(debug, "[INIT] use_tracking={}", use_tracking)

    tracker = None
    # выделяем из конфига нужные блоки
    models_cfg   = config["models"]
    pipeline_cfg = config["pipeline"]

    # инициализируем трекер
    tracker = DeepSort(**models_cfg["tracker"]) if use_tracking else None

    writer = None
    rand_palette = {}
    frame_count = 0
    width = height = None

    csv_rows: list = []

    try:
        for frame_bgr, meta in frame_gen:
            # VideoWriter
            if width is None:
                height, width = frame_bgr.shape[:2]
            if save_out and writer is None:
                fps = out_fps or meta.get("fps", 25)
                writer = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

            # зоны и линии
            if draw_regions:
                draw_zones(frame_bgr, zones)
                draw_lines(frame_bgr, lines)

            # детекция авто
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw = detect_cars(
                frame_rgb,
                **models_cfg["vehicle_detector"]
            )
            detections = [
                ([d["xmin"], d["ymin"], d["xmax"]-d["xmin"], d["ymax"]-d["ymin"]],
                 d["conf"], d["class_id"])
                for d in raw
            ]

            if use_tracking:
                tracks = tracker.update_tracks(detections, frame=frame_rgb)
                alive = set()
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    ltrb = track.to_ltrb(orig=True, orig_strict=True)
                    if ltrb is None:
                        continue

                    tid = track.track_id
                    alive.add(tid)
                    
                    # установка значения осей по умолчанию для нового трека
                    ensure_default_axles(tid)
                    # инициализация записей для csv
                    if tid not in TRACK_INFO:
                        init_track_info(tid, frame_count)

                    x1,y1,x2,y2 = map(int, ltrb)
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    info = TRACK_INFO[tid]

                    # классификация
                    crop, _, _ = crop_vehicle(frame_bgr, x1, y1, x2, y2, padding=0)
                    label = None
                    if crop is not None and crop.size:
                        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        _, label, _ = classify_car(
                            rgb_crop,
                            **models_cfg["classifier"]
                        )
                    info["class_id"] = label
                    info["last_frame"] = frame_count
                    info["length_frames"] += 1

                    # зоны
                    if zones:
                        zone = get_zone_for_point(cx, cy, zones)
                        if info["start_zone"] is None and zone:
                            info["start_zone"] = zone
                            info["enter_time"] = datetime.now().isoformat()
                        # если стартовая зона уже задана, то обновляем конечную
                        info["end_zone"] = zone
                        info["exit_time"] = datetime.now().isoformat()

                    # линии
                    if lines:
                        for name, (p1,p2) in lines.items():
                            prev = LINE_SIDE[tid].get(name, 0)
                            cur = side_of_line(cx, cy, p1, p2)
                            LINE_SIDE[tid][name] = cur
                            if crossed(prev, cur) and point_between_segment(cx, cy, p1, p2):
                                if info["start_line"] is None:
                                    info["start_line"] = name
                                    info["enter_time"] = datetime.now().isoformat()
                                info["end_line"] = name
                                info["exit_time"] = datetime.now().isoformat()

                    # оси
                    do_calc = (use_axle and should_calc_axle(
                        x2, y2, width, height,
                        True,
                            pipeline_cfg["hz_start"], pipeline_cfg["hz_stop"],
                            pipeline_cfg["vz_start"], pipeline_cfg["vz_stop"],
                    ) and (
                        tid not in LAST_AXLE_FRAME or 
                        frame_count - LAST_AXLE_FRAME[tid] >= pipeline_cfg.get("axle_period")
                    ))
                    if do_calc:
                        _, ax_count = process_wheels_on_vehicle(
                            frame_bgr, (x1, y1, x2, y2), **models_cfg["wheel_detector"]
                        )
                        update_axles(tid, ax_count)
                        info["axles"] = AXLE_VALUE.get(tid, DEFAULT_AXLES)
                        LAST_AXLE_FRAME[tid] = frame_count

                    stable = max(AXLE_VALUE.get(tid, DEFAULT_AXLES), DEFAULT_AXLES)
                    info["axles"] = stable
                    
                    # отрисовка
                    color = rand_palette.setdefault(
                        tid,
                        (random.randint(50,200), random.randint(50,200), random.randint(50,200))
                    )
                    
                    text = f"ID:{tid}" if show_track_id else ""
                    if label:
                        text += f" {label}"
                    if use_axle:
                        text += f" {stable}"
                    draw_box(frame_bgr, x1, y1, x2, y2, color, text=text)

                # финализировать неактивные
                for tid in list(TRACK_INFO):
                    if tid not in alive:
                        if TRACK_INFO[tid]["exit_time"] is None:
                            TRACK_INFO[tid]["exit_time"] = datetime.now().isoformat()
                        row = finalize_track(tid)
                        if row:
                            csv_rows.append(row)

            else:
                # только детекция
                for (x, y, w, h), conf, cls_id in detections:
                    x2, y2 = x + w, y + h
                    crop_img, _, _ = crop_vehicle(frame_bgr, x1, y1, x2, y2, padding=0)
                    label = None
                    if crop_img is not None and crop_img.size != 0:
                        rgb_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                        _, label, _ = classify_car(rgb_crop_img)
                    color = (255, 0, 0)
                    text = label or f"cls:{cls_id}"
                    if use_axle:
                        t, _ = process_wheels_on_vehicle(frame_bgr, (x,y,x2,y2), **models_cfg["wheel_detector"])
                        text += f" {t}"
                    draw_box(frame_bgr, x, y, x2, y2, color, text=text)

            # показать/сохранить
            if frame_count == 0:
                cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
            disp_w = 960
            disp_h = int(frame_bgr.shape[0]*disp_w/frame_bgr.shape[1])
            cv2.imshow("Tracking", cv2.resize(
                frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA
            ))
            if save_out and writer:
                writer.write(frame_bgr)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break

            frame_count += 1
            if not debug and frame_count % log_every == 0:
                print(f"[INFO] Processed {frame_count} frames")

    except KeyboardInterrupt:
        print("[EXIT] Interrupted by user")

    finally:
        # завершаем оставшиеся
        for tid in list(TRACK_INFO):
            if TRACK_INFO[tid]["exit_time"] is None:
                TRACK_INFO[tid]["exit_time"] = datetime.now().isoformat()
            row = finalize_track(tid)
            if row:
                csv_rows.append(row)

        # сохраняем CSV
        if csv_path and csv_rows:
            dbg(debug, "[CSV] Saved {} rows to {}", len(csv_rows), csv_path)
            write_csv(csv_path, CSV_HEADER, csv_rows)

        # очистка
        if save_out and writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[END] Total frames processed: {frame_count}")

def deepsort_sff(reader: SFFReader, **kwargs):
    """Запуск обработки SFFReader."""
    return _run_core(frame_gen=frames_from_sff(reader), **kwargs)

def deepsort_video(video_path: str, **kwargs):
    """Запуск обработки видеофайла."""
    return _run_core(frame_gen=frames_from_video(video_path), **kwargs)

