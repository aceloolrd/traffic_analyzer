import csv
from typing import List, Any

# Общий заголовок CSV для треков
CSV_HEADER: List[str] = [
    "track_id","class_id","axles",
    "start_zone","end_zone",
    "start_line","end_line",
    "first_frame","last_frame","length_frames",
    "enter_time","exit_time"
]

def write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    """
    Сохраняет таблицу в CSV-файл.
    :param path: путь к файлу
    :param header: список имён столбцов
    :param rows: список строк (каждая — список значений)
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
