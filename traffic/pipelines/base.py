# pipelines/base.py

from abc import ABC, abstractmethod
from typing import List, Tuple
from PIL import Image
import csv

class BaseRecognitionPipeline(ABC):
    def __init__(self, config: dict, parent=None):
        self.config = config
        # self.parent = parent
        # self.sign_data = []
        # self.train_model_data = []
        # self.skip_frames_count = 0
        # self.empty_frames_count = 0
        # self.max_rec_distance = config.get("max_rec_distance")
        # self.max_horizontal_distance = config.get("max_horizontal_distance")
        # self.picket_thr = config.get("picket_thr")

    # @abstractmethod
    # def detect(self, frame: Image.Image) -> List[dict]:
    #     pass

    # @abstractmethod
    # def classify(self, image: Image.Image) -> Tuple[int, str, float]:
    #     pass

    # @abstractmethod
    # def recognize_text(self, image: Image.Image) -> str:
    #     pass

    def _process_frame(self, frame: Image.Image, frame_num: int):
        # dets = self.detect(frame)
        # return frame, dets
        pass

    def _get_object_data(self, frame: Image.Image, dets: List[dict], frame_num: int):
        # for det in dets:
        #     xmin, ymin, xmax, ymax = [int(det[k]) for k in ("xmin","ymin","xmax","ymax")]
        #     crop = frame.crop((xmin,ymin,xmax,ymax))
        #     idx, label, conf = self.classify(crop)
        #     text = self.recognize_text(crop)
        #     self.sign_data.append({
        #         "frame": frame_num,
        #         "class_idx": idx,
        #         "label": label,
        #         "conf": conf,
        #         "text": text,
        #         "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        #     })
        pass

    def _save_data(self, output_csv_path: str):
        # with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        #     writer = csv.DictWriter(f, fieldnames=self.sign_data[0].keys())
        #     writer.writeheader()
        #     writer.writerows(self.sign_data)
        pass

    def process_video(self, frames, output_csv_path: str):
        # for i, frame in enumerate(frames):
        #     f, dets = self._process_frame(frame, i)
        #     self._get_object_data(f, dets, i)
        # self._save_data(output_csv_path)
        pass