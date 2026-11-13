import cv2
from ultralytics import YOLO
import numpy as np

class PersonDetector:
    def __init__(self, weights_path="yolov8n.pt"):
        self.model = YOLO(weights_path)

    def detect_persons(self, image_input, conf_threshold=0.5):
        """
        检测图片中所有人
        image_input: 可以是图片路径 (str) 或 numpy 数组 (np.ndarray)
        """
        # 判断输入类型
        if isinstance(image_input, str):
            # 图片路径
            results = self.model(image_input)[0]
            img = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            # 已经是 numpy 数组（视频帧）
            results = self.model(image_input)[0]
            img = image_input
        else:
            raise TypeError(f"不支持的输入类型: {type(image_input)}")
        
        persons = []
        boxes = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))
                # 裁剪人像
                person_crop = img[y1:y2, x1:x2]
                if person_crop.size > 0:
                    persons.append(person_crop)

        return persons, boxes