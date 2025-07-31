# classifiers/vehicle_classifier.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Tuple, List

                    
class VehicleClassifier:
    """
    Класс для классификации изображений автомобилей.

    Конфиг должен содержать:
      - model_path: str            # путь до весов .pth
      - class_names: List[str]     # метки классов
      - input_size: int            # размер (width==height) для CenterCrop
      - device: str                # 'cpu' или 'cuda:0'
      - half: bool                 # использовать FP16 (только на GPU)
    """

    def __init__(self, config: dict):
        self.model_path  = config["model_path"]
        self.class_names = config["class_names"]
        self.input_size  = config.get("input_size", 224)
        self.device      = config.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.half        = config.get("half", False) and "cuda" in self.device

        self._build_model()
        self._build_transform()

    def _build_model(self):
        # 1) Создаем ResNet34 backbone
        self.model = models.resnet34(weights=None)
        in_features   = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, len(self.class_names))

        # 2) Загружаем веса
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)

        # 3) Переводим на устройство
        self.model.to(self.device)
        if self.half:
            self.model.half()

        self.model.eval()

    def _build_transform(self):
        # препроцессинг совпадает с тем, на чем тренировалась модель
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def classify(self, image: Image.Image) -> Tuple[int, str, float]:
        """
        Классифицирует входное изображение.

        :param image: PIL.Image (RGB)
        :return: (class_idx, label, confidence)
        """
        # 1) Преобразуем PIL→Tensor и добавляем batch dimension
        x = self.transform(image).unsqueeze(0).to(self.device)
        if self.half:
            x = x.half()

        # 2) Прогоняем через модель
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            idx = idx.item()
            conf = conf.item()

        # 3) Возвращаем номер класса, метку и уверенность
        return idx, self.class_names[idx], conf
