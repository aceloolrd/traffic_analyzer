# detectors/vehicle_detector.py

from typing import List, Dict
from PIL import Image
import numpy as np
from ultralytics import YOLO

from detectors.base_detector import BaseDetector

class Detector(BaseDetector):

    def _load_model(self):
        self.model = YOLO(self.model_path)
        # self.model.fuse()


    def detect(self, frame: Image.Image) -> List[Dict]:
        img = np.array(frame)
        
        results = self.model(
            img,
            conf=self.config.get("confidence_threshold", 0.5),
            iou=self.config.get("iou_threshold", 0.5),
            classes=self.config.get("classes", None),
            max_det=self.config.get("max_det", 100),
            device=self.config.get("device", None),
            half=self.config.get("half", False),
            verbose=self.config.get("verbose", False)
        )[0]  
        
        dets: List[Dict] = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            dets.append({
                "bbox":       [x1, y1, x2, y2],
                "confidence": conf,
                "class_id":   cls
            })
            
        return dets

