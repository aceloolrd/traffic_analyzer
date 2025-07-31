# data_io/csv_writer.py

import os
import csv
from typing import List, Any, Optional

class CSVWriter:
    """
    Писатель CSV-таблиц для трекинговых данных.

    Класс содержит встроенный заголовок CSV_HEADER.
    """

    # Встроенный заголовок, используемый по умолчанию
    CSV_HEADER: List[str] = [
        "track_id", "class_id", "axles",
        "start_zone", "end_zone",
        "start_line", "end_line",
        "first_frame", "last_frame", "length_frames",
        "enter_time", "exit_time"
    ]

    def __init__(self, path: str, header: Optional[List[str]] = None):
        """
        :param path:   путь к выходному CSV-файлу
        :param header: список имён столбцов; если None, используется CSVWriter.CSV_HEADER
        """
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(self.path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)

        # Выбираем заголовок
        self._header = header or CSVWriter.CSV_HEADER
        # Записываем его сразу
        self._writer.writerow(self._header)
        self._file.flush()

    def write_row(self, row: List[Any]) -> None:
        """
        Записать одну строку.
        :param row: список значений в том же порядке, что и header.
        """
        if len(row) != len(self._header):
            raise ValueError(f"Row has {len(row)} elements but header has {len(self._header)} columns")
        self._writer.writerow(row)
        self._file.flush()

    def write_rows(self, rows: List[List[Any]]) -> None:
        """
        Записать несколько строк.
        :param rows: список строк
        """
        for row in rows:
            self.write_row(row)

    def close(self) -> None:
        """
        Закрыть файл. После этого записывать нельзя.
        """
        if not self._file.closed:
            self._file.close()
