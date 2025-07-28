import os

BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "weights"))

# модели
VEHICLE_DETECTOR = {
    "model_path": os.path.join(WEIGHTS_DIR, "yolo11m.pt"),
    "conf": 0.5,
    "iou": 0.5,
    "classes": [2, 3, 5, 7],
}
WHEEL_DETECTOR = {
    "model_path": os.path.join(WEIGHTS_DIR, "best0.pt"),
    "conf": 0.25,
    "iou": 0.5,
}
CLASSIFIER = {
    "model_path": os.path.join(WEIGHTS_DIR, "resnet34_vehicle_classifier_224.pth"),
    "class_names": [
        "bus", "moto", "tractor", "tractor-trailer",
        "truck", "truck-trailer",
    ],
    "input_size": 224,
}
TRACKER = {
    "max_iou_distance": 0.5,
    "max_age": 20,
    "n_init": 3,
    "max_cosine_distance": 0.2,
    "nn_budget": 100,
    "gating_only_position": True,
    "embedder": "mobilenet",
    "half": True,
    "bgr": False,
    "embedder_gpu": True,
}

# параметры пайплайна
PIPELINE = {
    "max_rec_distance": 5,
    "max_horizontal_distance": 2,
    "picket_thr": 2,
    "axle_period": 0,
    "hz_start": None,
    "hz_stop": None,
    "vz_start": None,
    "vz_stop": None,
}

# объединённый конфиг
CONFIG = {
    "models": {
        "vehicle_detector": VEHICLE_DETECTOR,
        "wheel_detector":   WHEEL_DETECTOR,
        "classifier":       CLASSIFIER,
        "tracker":          TRACKER,
    },
    "pipeline": PIPELINE,
}
