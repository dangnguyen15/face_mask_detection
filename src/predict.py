"""
predict.py
----------
Dùng mô hình Keras (.keras) để phân loại từng ROI khuôn mặt:
  - "Co khau trang"  (wearing mask)
  - "Khong co khau trang" (no mask)
"""
 
from __future__ import annotations
import numpy as np
import cv2
import zipfile
import tempfile
import os
 
IMG_SIZE = (128, 128)
LABELS = ["Co khau trang", "Khong co khau trang"]
COLORS = {
    "Co khau trang": (0, 200, 0),
    "Khong co khau trang": (0, 0, 220),
}
 
 
class MaskPredictor:
    def __init__(self, model_path: str):
        try:
            self.model = self._build_and_load(model_path)
        except Exception as e:
            raise RuntimeError(f"Khong the load mask model tu '{model_path}': {e}")
 
    def _build_and_load(self, model_path: str):
        from tensorflow import keras
        import h5py
 
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(model_path, "r") as zf:
            zf.extractall(tmp_dir)
        weights_path = os.path.join(tmp_dir, "model.weights.h5")
 
        model = keras.Sequential([
            keras.layers.Input(shape=(128, 128, 3)),
            keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2d"),
            keras.layers.MaxPooling2D((2, 2), name="max_pooling2d"),
            keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_1"),
            keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1"),
            keras.layers.Conv2D(128, (3, 3), activation="relu", name="conv2d_2"),
            keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_2"),
            keras.layers.Flatten(name="flatten"),
            keras.layers.Dense(128, activation="relu", name="dense"),
            keras.layers.Dropout(0.5, name="dropout"),
            keras.layers.Dense(1, activation="sigmoid", name="dense_1"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
 
        with h5py.File(weights_path, "r") as f:
            for layer in model.layers:
                name = layer.name
                # Key dung backslash trong ten: "layers\conv2d"
                layer_key = f"layers\\{name}"
                if layer_key not in f:
                    continue
                vars_group = f[layer_key]["vars"]
                weights = [vars_group[str(i)][()] for i in range(len(vars_group))]
                if weights:
                    layer.set_weights(weights)
                    print(f"[predict] Loaded {name}: std={weights[0].std():.4f}")
 
        print("[predict] Load weights thanh cong!")
        return model
 
    def preprocess(self, face_roi: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_roi, IMG_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype("float32") / 255.0
        return np.expand_dims(normalized, axis=0)
 
    def predict(self, face_roi: np.ndarray) -> tuple:
        if face_roi is None or face_roi.size == 0:
            return LABELS[1], 0.0
        tensor = self.preprocess(face_roi)
        pred = float(self.model.predict(tensor, verbose=0)[0][0])
        if pred >= 0.38:
            return LABELS[0], pred
        else:
            return LABELS[1], 1.0 - pred
 
    def predict_batch(self, face_rois: list) -> list:
        if not face_rois:
            return []
        return [self.predict(roi) for roi in face_rois]