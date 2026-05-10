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
    """
    Thread đọc frame từ webcam và chạy inference.
 
    Parameters
    ----------
    detector : YOLOFaceDetector
    predictor : MaskPredictor
    on_frame : callable(annotated_frame, stats_dict)
        Callback được gọi mỗi khi có frame mới đã xử lý.
        stats_dict = {"total": int, "mask": int, "no_mask": int, "fps": float}
    camera_index : int
        Index camera (mặc định 0 — webcam chính)
    target_fps : int
        FPS tối đa muốn xử lý (để giảm tải CPU/GPU)
    """
 
    def __init__(
        self,
        detector: YOLOFaceDetector,
        predictor: MaskPredictor,
        on_frame,
        camera_index: int = 0,
        target_fps: int = 20,
    ):
        super().__init__(daemon=True)
        self.detector = detector
        self.predictor = predictor
        self.on_frame = on_frame
        self.camera_index = camera_index
        self.target_fps = target_fps
 
        self._running = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # bắt đầu ở trạng thái "chạy"
 
        self.cap: cv2.VideoCapture | None = None
 
    # ------------------------------------------------------------------
    # Public controls
    # ------------------------------------------------------------------
 
    def stop(self):
        """Dừng thread và giải phóng camera."""
        self._running.clear()
 
    def pause(self):
        self._paused.clear()
 
    def resume(self):
        self._paused.set()
 
    @property
    def is_running(self) -> bool:
        return self._running.is_set()
 
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
 
    def run(self):
        self._running.set()
        self.cap = cv2.VideoCapture(self.camera_index)
 
        if not self.cap.isOpened():
            print(f"[CameraThread] Không thể mở camera index={self.camera_index}")
            self._running.clear()
            return
 
        # Tuỳ chọn: tăng độ phân giải nếu webcam hỗ trợ
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
        interval = 1.0 / self.target_fps
        prev_time = time.time()
 
        while self._running.is_set():
            self._paused.wait()  # block nếu đang pause
 
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
 
            # Giới hạn FPS xử lý
            now = time.time()
            elapsed = now - prev_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
            fps = 1.0 / max(time.time() - prev_time, 1e-6)
            prev_time = time.time()
 
            annotated, stats = self._process_frame(frame, fps)
            self.on_frame(annotated, stats)
 
        if self.cap:
            self.cap.release()
 
    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
 
    def _process_frame(
        self, frame: np.ndarray, fps: float
    ) -> tuple[np.ndarray, dict]:
        """Detect faces → classify masks → annotate frame."""
        boxes = self.detector.detect(frame)
 
        mask_count = 0
        no_mask_count = 0
 
        if boxes:
            # Lấy ROI cho từng khuôn mặt
            rois = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes]
            predictions = self.predictor.predict_batch(rois)
 
            for (x1, y1, x2, y2), (label, conf) in zip(boxes, predictions):
                color = COLORS.get(label, (200, 200, 200))
 
                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
 
                # Nhãn + confidence
                text = f"{label}: {conf:.0%}"
                (tw, th), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1
                )
                # Nền cho chữ
                cv2.rectangle(
                    frame,
                    (x1, y1 - th - baseline - 6),
                    (x1 + tw + 4, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    text,
                    (x1 + 2, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
 
                if label == "Co khau trang":
                    mask_count += 1
                else:
                    no_mask_count += 1
 
        # HUD: FPS góc trên phải
        h, w = frame.shape[:2]
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (w - 110, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
 
        stats = {
            "total": len(boxes),
            "mask": mask_count,
            "no_mask": no_mask_count,
            "fps": fps,
        }
        return frame, stats