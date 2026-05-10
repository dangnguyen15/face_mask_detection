"""
yolo_detector.py
----------------
Dùng YOLOv3 (file .pt từ ultralytics/torch) để phát hiện khuôn mặt người.
Trả về danh sách bounding-box (x1, y1, x2, y2) cho mỗi khuôn mặt tìm thấy.
"""
 
import numpy as np
import torch
 
 
class YOLOFaceDetector:
    """
    Wrapper quanh mô hình YOLOv3 (.pt) để phát hiện khuôn mặt.
 
    Parameters
    ----------
    model_path : str
        Đường dẫn tới file yolov3.pt
    conf_threshold : float
        Ngưỡng confidence tối thiểu (mặc định 0.5)
    device : str | None
        'cuda', 'cpu', hoặc None (tự chọn)
    """
 
    def __init__(self, model_path: str, conf_threshold: float = 0.5, device: str | None = None):
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
 
        # Load model bằng torch.hub (ultralytics yolov5/v3 compatible)
        # Nếu file .pt là custom trained thì dùng custom=True
        try:
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=model_path,
                force_reload=False,
                device=self.device,
                verbose=False,
            )
            self.model.conf = conf_threshold
        except Exception as e:
            raise RuntimeError(f"Không thể load YOLOv3 model từ '{model_path}': {e}")
 
    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Phát hiện khuôn mặt trong một frame BGR (numpy array từ OpenCV).
 
        Returns
        -------
        list of (x1, y1, x2, y2)  — toạ độ pixel, đã clip vào kích thước frame
        """
        if frame is None or frame.size == 0:
            return []
 
        # YOLOv5/v3 hub model nhận BGR numpy array trực tiếp
        results = self.model(frame)
 
        boxes: list[tuple[int, int, int, int]] = []
        h, w = frame.shape[:2]
 
        # results.xyxy[0] → tensor shape (N, 6): x1 y1 x2 y2 conf cls
        detections = results.xyxy[0].cpu().numpy()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf < self.conf_threshold:
                continue
            x1 = int(np.clip(x1, 0, w))
            y1 = int(np.clip(y1, 0, h))
            x2 = int(np.clip(x2, 0, w))
            y2 = int(np.clip(y2, 0, h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
 
        return boxes
 