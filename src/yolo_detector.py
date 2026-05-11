import cv2
import numpy as np
 
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("[Detector] mediapipe chưa được cài. Chạy: pip install mediapipe")
 
import urllib.request
import os
 
# Model nhỏ tự tải về lần đầu (~800 KB)
_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")
 
 
def _ensure_model():
    """Tải model tflite về nếu chưa có."""
    if os.path.exists(_MODEL_PATH):
        return
    print("[Detector] Đang tải MediaPipe face model (~800 KB)...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[Detector] Tải xong.")
 
 
class YOLOFaceDetector:
    """
    Wrapper MediaPipe FaceDetector (Tasks API - v0.10+).
    Giữ tên class để camera.py không cần đổi import.
 
    Trả về list: [x1, y1, x2, y2, confidence]
    """
 
    def __init__(
        self,
        model_path: str = "",        # không dùng, giữ tương thích
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45, # không dùng
        model_selection: int = 1,
    ):
        if not _MP_AVAILABLE:
            raise RuntimeError("MediaPipe chưa được cài!\nChạy: pip install mediapipe")
 
        self.conf_threshold = conf_threshold
 
        _ensure_model()
 
        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=conf_threshold,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)
        print(f"[Detector] MediaPipe FaceDetector (Tasks API) sẵn sàng  conf={conf_threshold}")
 
    def detect(self, frame: np.ndarray) -> list:
        """
        frame: BGR numpy array
        returns: [[x1, y1, x2, y2, confidence], ...]
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._detector.detect(mp_image)
 
        boxes = []
        for det in result.detections:
            score = det.categories[0].score
            if score < self.conf_threshold:
                continue
 
            bb = det.bounding_box
            x1 = max(0, bb.origin_x)
            y1 = max(0, bb.origin_y)
            x2 = min(w, bb.origin_x + bb.width)
            y2 = min(h, bb.origin_y + bb.height)
 
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
 
            boxes.append([x1, y1, x2, y2, float(score)])
 
        return boxes
 
    def close(self):
        self._detector.close()