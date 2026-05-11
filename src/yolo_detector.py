"""
yolo_detector.py
----------------
Dùng YOLOv3 (file .pt từ ultralytics/torch) để phát hiện khuôn mặt người.
Trả về danh sách bounding-box (x1, y1, x2, y2) cho mỗi khuôn mặt tìm thấy.
"""
 
from __future__ import annotations
import numpy as np
import cv2
import shutil
import os
import tempfile
 
 
class YOLOFaceDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5, device=None):
        self.conf_threshold = conf_threshold
 
        # Copy xml ra thu muc tam tranh loi encoding tieng Viet
        tmp = tempfile.gettempdir()
 
        frontal_src = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        alt_src     = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        profile_src = cv2.data.haarcascades + "haarcascade_profileface.xml"
 
        frontal_dst = os.path.join(tmp, "hc_frontal.xml")
        alt_dst     = os.path.join(tmp, "hc_alt2.xml")
        profile_dst = os.path.join(tmp, "hc_profile.xml")
 
        shutil.copy2(frontal_src, frontal_dst)
        shutil.copy2(alt_src,     alt_dst)
        shutil.copy2(profile_src, profile_dst)
 
        self.cascade_frontal = cv2.CascadeClassifier(frontal_dst)
        self.cascade_alt     = cv2.CascadeClassifier(alt_dst)
        self.cascade_profile = cv2.CascadeClassifier(profile_dst)
 
        if self.cascade_frontal.empty():
            raise RuntimeError("Khong load duoc cascade frontal")
        print("[YOLOFaceDetector] Haar Cascade (frontal + alt + profile) san sang.")
 
    def detect(self, frame: np.ndarray) -> list:
        if frame is None or frame.size == 0:
            return []
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = frame.shape[:2]
 
        boxes_raw = []
 
        # Cascade 1: frontal default
        faces1 = self.cascade_frontal.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
        if len(faces1) > 0:
            boxes_raw.extend(faces1.tolist())
 
        # Cascade 2: frontal alt2 (tot hon khi deo khau trang)
        faces2 = self.cascade_alt.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
        if len(faces2) > 0:
            boxes_raw.extend(faces2.tolist())
 
        # Cascade 3: profile (mat nghieng)
        faces3 = self.cascade_profile.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
        if len(faces3) > 0:
            boxes_raw.extend(faces3.tolist())
 
        if not boxes_raw:
            return []
 
        # Gop cac box trung nhau bang NMS don gian
        boxes = self._nms(boxes_raw, h, w)
        return boxes
 
    def _nms(self, boxes_raw: list, h: int, w: int) -> list:
        """Non-maximum suppression don gian de gop box trung."""
        if not boxes_raw:
            return []
 
        # Chuyen sang (x1,y1,x2,y2)
        rects = []
        for (x, y, fw, fh) in boxes_raw:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + fw)
            y2 = min(h, y + fh)
            if x2 > x1 and y2 > y1:
                rects.append([x1, y1, x2, y2])
 
        if not rects:
            return []
 
        # Dung groupRectangles cua OpenCV
        rects_for_group = []
        for (x1, y1, x2, y2) in rects:
            rects_for_group.append([x1, y1, x2 - x1, y2 - y1])
            rects_for_group.append([x1, y1, x2 - x1, y2 - y1])  # can duplicate
 
        grouped, _ = cv2.groupRectangles(rects_for_group, 1, 0.3)
 
        if len(grouped) == 0:
            # Neu groupRectangles xoa het, tra ve rects goc
            return [(x1, y1, x2, y2) for (x1, y1, x2, y2) in rects]
 
        result = []
        for (x, y, fw, fh) in grouped:
            result.append((int(x), int(y), int(x+fw), int(y+fh)))
        return result