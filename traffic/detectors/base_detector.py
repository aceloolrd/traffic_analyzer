# detectors/base_detector.py

from abc import ABC, abstractmethod
from typing import List, Dict
from PIL import Image

class BaseDetector(ABC):
    """
    Абстрактный класс для детекторов.
    Конфиг должен содержать:
      - model_path (str, опционально)
      - device (str, опционально)
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_path = config.get("model_path", None)
        self.device = config.get("device", "0")
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """
        Загрузка/инициализация модели.
        Вызывается при создании экземпляра.
        """
        ...

    @abstractmethod
    def detect(self, frame: Image.Image) -> List[Dict]:
        """
        Детекция объектов на кадре.
        :param frame: PIL.Image
        :return: список словарей с ключами:
            - bbox: [xmin, ymin, xmax, ymax]
            - confidence: float
            - class_id: int или None
        """
        ...