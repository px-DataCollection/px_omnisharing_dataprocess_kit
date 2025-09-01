import time
from ultralytics import YOLO
import os
import cv2

class yolo():
    def __init__(self, model_path=None):
        current_dir = os.getcwd()
        self.model = YOLO(
            model_path,
            task="segment")
        self.cnt = 2
        
    def predict(self, image, conf=0.7):
        # now = time.time()
        results = self.model(image, show=False, save=False, conf=conf, verbose=False,imgsz = [704,960])[0]
        self.cnt += 1
        detected = []
        imgsz = image.shape[:2]
        if results.masks is not None:
            boxes = results.boxes
            masks = results.masks.cpu().numpy()
            for i in range(masks.shape[0]):
                detected.append({
                    'cls': int(boxes.cls[i].cpu()),
                    'bbox_cxywh': boxes.xywh[i].cpu().numpy(),
                    'mask': masks[i].data.reshape(imgsz[0], imgsz[1]),
                    'det_conf': boxes.conf[i].cpu().numpy(),
                    'obj_id': int(results.names[int(boxes.cls[i].cpu())].split('_')[-1]
                                  if results.names[int(boxes.cls[i].cpu())].split('_')[-1] != 'none' else -1, ),
                    'name': results.names[int(boxes.cls[i].cpu())]
                })
        return detected
