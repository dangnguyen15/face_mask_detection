"""
predict.py
----------
Dùng mô hình Keras (.keras) để phân loại từng ROI khuôn mặt:
  - "Co khau trang"  (wearing mask)
  - "Khong co khau trang" (no mask)
"""
 
import numpy as np
import cv2
from tensorflow.keras.models import load_model  # type: ignore
 
 
# Kích thước ảnh đầu vào mô hình (phải khớp với lúc bạn train)
IMG_SIZE = (224, 224)
 
# Nhãn — thứ tự phải đúng với class index khi train
LABELS = ["Co khau trang", "Khong co khau trang"]
 
# Màu bounding-box tương ứng (BGR cho OpenCV)
COLORS = {
    "Co khau trang": (0, 200, 0),       # xanh lá
    "Khong co khau trang": (0, 0, 220), # đỏ
}
 
 
class MaskPredictor:
    """
    Phân loại khuôn mặt có đeo khẩu trang hay không.
 
    Parameters
    ----------
    model_path : str
        Đường dẫn tới face_mask_model.keras
    """
 
    def __init__(self, model_path: str):
        try:
            self.model = load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Không thể load mask model từ '{model_path}': {e}")
 
    def preprocess(self, face_roi: np.ndarray) -> np.ndarray:
        """Resize + normalize ROI về tensor (1, H, W, 3)."""
        resized = cv2.resize(face_roi, IMG_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype("float32") / 255.0
        return np.expand_dims(normalized, axis=0)  # (1, 224, 224, 3)
 
    def predict(self, face_roi: np.ndarray) -> tuple[str, float]:
        """
        Dự đoán nhãn và độ tin cậy cho một khuôn mặt.
 
        Returns
        -------
        (label, confidence)  —  label: str, confidence: float [0,1]
        """
        if face_roi is None or face_roi.size == 0:
            return LABELS[1], 0.0
 
        tensor = self.preprocess(face_roi)
        preds = self.model.predict(tensor, verbose=0)[0]  # shape (num_classes,)
 
        idx = int(np.argmax(preds))
        label = LABELS[idx] if idx < len(LABELS) else "Khong xac dinh"
        confidence = float(preds[idx])
        return label, confidence
 
    def predict_batch(
        self, face_rois: list[np.ndarray]
    ) -> list[tuple[str, float]]:
        """Dự đoán nhiều khuôn mặt cùng lúc (hiệu quả hơn loop)."""
        if not face_rois:
            return []
 
        batch = np.concatenate([self.preprocess(roi) for roi in face_rois], axis=0)
        all_preds = self.model.predict(batch, verbose=0)  # (N, num_classes)
 
        results = []
        for preds in all_preds:
            idx = int(np.argmax(preds))
            label = LABELS[idx] if idx < len(LABELS) else "Khong xac dinh"
            results.append((label, float(preds[idx])))
        return results