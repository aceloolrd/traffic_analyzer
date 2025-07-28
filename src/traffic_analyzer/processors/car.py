import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Dict, List, Tuple

# Кэш моделей, чтобы не инициализировать их каждый кадр
_detectors: Dict[str, YOLO] = {}
_classifiers: Dict[str, nn.Module] = {}

# Препроцесс для классификации (фиксируем сюда normalize+crop, размер будем передавать)
_base_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop,  # сюда будет передан размер в вызове
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

def detect_cars(
    frame: np.ndarray,
    model_path: str,
    conf: float,
    iou: float,
    classes: List[int],
    half: bool = True,
    device: str = '0',
    max_det: int = 20,
    verbose: bool = False,
) -> List[Dict]:
    """
    Детектирует транспортные средства с помощью YOLO.
    Все параметры (путь к весам, пороги и т.п.) передаются в аргументах.
    """
    # ленивый инит детектора
    if model_path not in _detectors:
        _detectors[model_path] = YOLO(model_path)
    detector = _detectors[model_path]

    results = detector(
        frame,
        conf=conf,
        iou=iou,
        classes=classes,
        half=half,
        device=device,
        max_det=max_det,
        verbose=verbose
    )[0]

    dets: List[Dict] = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        dets.append({
            "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
            "conf": float(box.conf[0]), "class_id": int(box.cls[0])
        })
    return dets


def classify_car(
    crop: np.ndarray,
    model_path: str,
    class_names: List[str],
    input_size: int,
    device: str = None,
) -> Tuple[int, str, float]:
    """
    Классифицирует кадр (crop) в один из классов.
    Параметры: путь к весам, список имен классов, требуемый input_size.
    """
    # определяем устройство
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # ленивый инит классификатора
    if model_path not in _classifiers:
        # создаём ResNet34
        clf = models.resnet34(weights=None)
        in_feats = clf.fc.in_features
        clf.fc = nn.Linear(in_feats, len(class_names))
        clf.load_state_dict(torch.load(model_path, map_location=dev))
        clf.to(dev).eval()
        _classifiers[model_path] = clf
    else:
        clf = _classifiers[model_path]

    # строим препроцесс с нужным CenterCrop
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    inp = preprocess(crop).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = clf(inp)
    idx = int(out.argmax(1).item())
    prob = float(torch.softmax(out, dim=1)[0, idx].item())
    return idx, class_names[idx], prob
