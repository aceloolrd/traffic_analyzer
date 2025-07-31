# data_io/sff_reader.py

import struct
from PIL import Image
from typing import Optional, Tuple, Generator, Dict
import io

import cv2
import numpy as np

"""
Модуль: sff_reader.py

Содержит класс SFFReader для работы с видео в формате .sff/.dat:
- Чтение заголовка .sff
- Чтение индексов кадров из .dat
- Получение кадров по номеру или по значению пикета
- Генерация кадров с пропуском близких по пикету
- Пересчет значений пикетов при рассогласовании заголовка и индекса
"""

class SFFReader:
    """
    Класс для чтения и обработки данных из файлов формата SFF.

    Атрибуты:
        sff_file (str): путь к файлу .sff.
        dat_file (str): путь к файлу .dat для индексов.
        header (dict): информация из заголовка .sff.
        frame_data (list): список метаданных по каждому кадру.
        frame_count (int): количество кадров.
    """

    def __init__(self, sff_file: str, dat_file: str = None):
        """
        Инициализация SFFReader.

        Аргументы:
            sff_file (str): путь к файлу .sff.
            dat_file (str, optional): путь к файлу .dat. Если не указан,
                добавляется расширение .dat к имени .sff.
        """
        self.sff_file = sff_file
        self.dat_file = dat_file or sff_file.replace(".sff", ".dat")
        self.header = self._read_header()
        self.frame_data = self._read_dat()
        self.frame_count = len(self.frame_data)
        self.recalculate_pickets()

    def _read_header(self) -> dict:
        """
        Чтение и разбор заголовка файла .sff.

        Возвращает:
            dict: поля заголовка:
                'road_code', 'road_name_length', 'road_name',
                'direction', 'recording_date', 'start_km', 'end_km', 'reserved'
        """
        with open(self.sff_file, "rb") as f:
            header = {
                "road_code": struct.unpack("<h", f.read(2))[0],
                "road_name_length": struct.unpack("<h", f.read(2))[0],
                "road_name": f.read(200).decode("cp1251").rstrip("\x00"),
                "direction": struct.unpack("<h", f.read(2))[0],
                "recording_date": struct.unpack("<d", f.read(8))[0],
                "start_km": struct.unpack("<d", f.read(8))[0],
                "end_km": struct.unpack("<d", f.read(8))[0],
                "reserved": f.read(5),
            }
        return header

    def _read_dat(self) -> list:
        """
        Чтение индекса кадров из DAT-файла.

        Возвращает:
            list: список словарей с полями:
                'jpeg_size', 'offset', 'picket', 'timestamp'
        """
        frame_data = []
        with open(self.dat_file, "rb") as f_dat:
            while True:
                dat_data = f_dat.read(24)
                if len(dat_data) < 24:
                    break
                jpeg_size, offset, picket, timestamp = struct.unpack("<iqid", dat_data)
                frame_data.append({
                    "jpeg_size": jpeg_size,
                    "offset": offset,
                    "picket": picket,
                    "timestamp": timestamp,
                })
        return frame_data

    def get_frames_count(self) -> int:
        """
        Получить общее количество кадров.

        Возвращает:
            int: количество кадров.
        """
        return self.frame_count

    def get_frame_by_number(self, frame_number: int, as_bytes: bool = False):
        """
        Извлечь кадр по его номеру.

        Аргументы:
            frame_number (int): индекс кадра (0..frame_count-1).
            as_bytes (bool): если True, вернуть JPEG-байты, иначе PIL.Image.

        Возвращает:
            bytes или PIL.Image: данные кадра.
        """
        if frame_number >= self.frame_count:
            raise ValueError(f"Номер кадра {frame_number} вне диапазона.")
        info = self.frame_data[frame_number]
        with open(self.sff_file, "rb") as f:
            f.seek(info["offset"])
            data = f.read(info["jpeg_size"]).rstrip(b"\x00")
        if as_bytes:
            return data
        return Image.open(io.BytesIO(data))

    def get_frame_by_meter(self, meter: int) -> Tuple[Optional[Image.Image], int]:
        """
        Найти и вернуть кадр, ближайший к заданному значению пикета.

        Аргументы:
            meter (int): желаемый пикет.

        Возвращает:
            Tuple[Optional[PIL.Image], int]: кадр и его номер;
            (None, -1) если не найдено.
        """
        closest = min(
            enumerate(self.frame_data),
            key=lambda x: abs(x[1]["picket"] - meter),
            default=(None, None),
        )
        idx, _ = closest
        if idx is None:
            return None, -1
        return self.get_frame_by_number(idx), idx

    def get_frames(
        self,
        direction: int = 0,
        start_km: float = None,
        end_km: float = None,
        reverse: bool = False,
        picket_thr: int = 2,
    ) -> Generator[Tuple[int, Image.Image, int, float], None, None]:
        """
        Генерация кадров с фильтрацией по пикету.

        Аргументы:
            direction (int): направление (0 или 1).
            start_km (float): начало отрезка в километрах.
            end_km (float): конец отрезка в километрах.
            reverse (bool): инвертировать порядок кадров.
            picket_thr (int): порог в метрах для пропуска близких кадров.

        Возвращает:
            Генератор кортежей (frame_idx, image, picket, timestamp).
        """
        prev_picket = None
        skipped = False
        indices = list(range(self.frame_count))
        if reverse:
            indices.reverse()
        with open(self.sff_file, "rb") as f:
            for i in indices:
                info = self.frame_data[i]
                if prev_picket is not None:
                    diff = abs(info["picket"] - prev_picket)
                    if diff < picket_thr and not skipped:
                        skipped = True
                        continue
                    skipped = False
                prev_picket = info["picket"]
                f.seek(info["offset"])
                data = f.read(info["jpeg_size"]).rstrip(b"\x00")
                img = Image.open(io.BytesIO(data))
                yield i, img, info["picket"], info["timestamp"]

    def recalculate_pickets(self):
        """
        Коррекция значений пикетов при рассогласовании заголовка и DAT.

        Логика:
            - Если header.start_km/end_km == 0, выставить из DAT.
            - Иначе интерполировать новые пикеты по диапазону.
        """
        start_h = self.header.get("start_km", 0) * 1000
        end_h = self.header.get("end_km", 0) * 1000
        start_d = self.frame_data[0]["picket"]
        end_d = self.frame_data[-1]["picket"]
        if start_h == 0 and end_h == 0:
            self.header["start_km"] = round(start_d / 1000, 3)
            self.header["end_km"] = round(end_d / 1000, 3)
        if start_h != start_d or end_h != end_d:
            for fr in self.frame_data:
                poff = (end_h - start_h) * (fr["picket"] - start_d) / (end_d - start_d)
                fr["picket"] = int(round(start_h + poff))



def frames_from_sff(reader: SFFReader) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """
    Генератор кадров из SFFReader (конвертация PIL→BGR).

    :param reader: экземпляр SFFReader
    :yield: (frame, info), где
        frame: np.ndarray (BGR),
        info: dict с keys ['frame_num', 'picket', 'timestamp']
    """
    for idx, img_pil, picket, timestamp in reader.get_frames():
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        yield frame, {
            "frame_num": idx,
            "picket": picket,
            "timestamp": timestamp
        }
