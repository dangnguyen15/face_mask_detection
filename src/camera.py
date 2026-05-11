"""
camera.py
---------
Quản lý vòng lặp capture từ webcam, chạy trong thread riêng.
Kết hợp YOLO detector + Mask predictor để xử lý từng frame,
rồi trả frame đã annotate qua callback.
"""
 
from __future__ import annotations
import threading
import time
import cv2
import numpy as np
 
from src.yolo_detector import YOLOFaceDetector
from src.predict import MaskPredictor, COLORS
 
 
class CameraThread(threading.Thread):
    def __init__(self, detector: YOLOFaceDetector, predictor: MaskPredictor,
                 on_frame, camera_index: int = 0, target_fps: int = 25):
        super().__init__(daemon=True)
        self.detector = detector
        self.predictor = predictor
        self.on_frame = on_frame
        self.camera_index = camera_index
        self.target_fps = target_fps
 
        self._running = threading.Event()
        self._paused = threading.Event()
        self._paused.set()
        self.cap = None
 
        # Cache: chi chay mask predict moi N frame de giam lag
        self._frame_count = 0
        self._predict_every = 2   # predict 1 lan moi 2 frame
        self._last_boxes = []
        self._last_preds = []
 
    @property
    def is_running(self):
        return self._running.is_set()
 
    def stop(self):
        self._running.clear()
 
    def run(self):
        self._running.set()
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW nhanh hon tren Windows
 
        if not self.cap.isOpened():
            print(f"[CameraThread] Khong the mo camera index={self.camera_index}")
            self._running.clear()
            return
 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # giam buffer de giam do tre
 
        interval = 1.0 / self.target_fps
        prev_time = time.time()
 
        while self._running.is_set():
            self._paused.wait()
 
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue
 
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
 
            annotated, stats = self._process_frame(frame, fps)
            self.on_frame(annotated, stats)
 
            # Gioi han FPS
            elapsed = time.time() - prev_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
 
        if self.cap:
            self.cap.release()
 
    def _process_frame(self, frame: np.ndarray, fps: float):
        self._frame_count += 1
 
        # Chi detect + predict moi N frame
        if self._frame_count % self._predict_every == 0:
            boxes = self.detector.detect(frame)
            self._last_boxes = boxes
 
            if boxes:
                rois = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes]
                self._last_preds = self.predictor.predict_batch(rois)
            else:
                self._last_preds = []
 
        boxes = self._last_boxes
        predictions = self._last_preds
 
        mask_count = 0
        no_mask_count = 0
 
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if i >= len(predictions):
                break
            label, conf = predictions[i]
            color = COLORS.get(label, (200, 200, 200))
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
 
            text = f"{label}: {conf:.0%}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - bl - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - bl - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
 
            if label == "Co khau trang":
                mask_count += 1
            else:
                no_mask_count += 1
 
        # FPS
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS:{fps:.0f}", (w - 90, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 2, cv2.LINE_AA)
 
        stats = {
            "total": len(boxes),
            "mask": mask_count,
            "no_mask": no_mask_count,
            "fps": fps,
        }
        return frame, stats
 