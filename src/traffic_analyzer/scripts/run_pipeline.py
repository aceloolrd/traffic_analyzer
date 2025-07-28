"""
Модуль: run_pipeline.py

Скрипт для запуска обработки видео:
- SFFReader + DeepSort (deepsort_reader)
- Обычные видеофайлы + DeepSort (deepsort_video)

Использует конфигурацию PIPELINE из config.py.
"""

from traffic_analyzer.readers.sff_reader import SFFReader
from traffic_analyzer.pipelines.main_pipeline import deepsort_sff, deepsort_video
from traffic_analyzer.config import CONFIG

def main():
    """
    Точка входа: пример запуска для SFF и (закомментированного) видео.
    """
    # Путь к видеофайлу и .sff
    video_path = r"C:\Users\acehm\OneDrive\Рабочий стол\baseline\base\ultralytics.mp4"
    sff_path = r"E:\Camera_777\Video.sff"

    # Обработка SFF
    reader = SFFReader(sff_path)
    deepsort_sff(
        reader=reader,
        config=CONFIG,
        save_out=False,
        # csv_path="counts_from_sff.csv",
        use_tracking=True,
        use_axle=True,
        draw_regions=True,
        # zones={'ZONE1': [(x1, y1), …], …},
        
        lines = {
            "LINE_A": ((1900,400), (1900,1600)),
            "LINE_B": ((10,200), (10,1000)),
}
    )

    # Обработка обычного видео (раскомментировать при необходимости)
    # deepsort_video(
    #     video_path=video_path,
    #     config=CONFIG,
    #     save_out=True,
    #     out_path="out_from_mp4.mp4",
    #     csv_path="counts_from_mp4.csv",
    #     use_tracking=True,
    #     use_axle=True,
    #     draw_regions=True,
    # )
    
if __name__ == "__main__":
    main()
