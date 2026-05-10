"""
main.py
-------
Điểm khởi động của ứng dụng nhận diện khẩu trang realtime.
 
Cách chạy:
    python main.py
 
Yêu cầu thư viện:
    pip install torch torchvision ultralytics tensorflow opencv-python pillow
"""
 
import tkinter as tk
import sys
import os
 
# ── Đảm bảo import từ thư mục gốc ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
 
from src.yolo_detector import YOLOFaceDetector
from src.predict import MaskPredictor
from src.camera import CameraThread
from src.ui import MaskDetectionUI
 
# ── Đường dẫn model ─────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = os.path.join(ROOT, "model", "yolov3.pt")
MASK_MODEL_PATH = os.path.join(ROOT, "model", "face_mask_model.keras")
 
 
def main():
    # 1. Load models (thực hiện 1 lần khi khởi động)
    print("[main] Đang load YOLOv3 model...")
    try:
        detector = YOLOFaceDetector(YOLO_MODEL_PATH, conf_threshold=0.5)
    except RuntimeError as e:
        print(f"[LỖI] {e}")
        sys.exit(1)
 
    print("[main] Đang load Mask Keras model...")
    try:
        predictor = MaskPredictor(MASK_MODEL_PATH)
    except RuntimeError as e:
        print(f"[LỖI] {e}")
        sys.exit(1)
 
    print("[main] Cả hai model đã sẵn sàng.")
 
    # 2. Factory tạo CameraThread mới (dùng khi user nhấn "Bắt đầu")
    def camera_factory() -> CameraThread:
        return CameraThread(
            detector=detector,
            predictor=predictor,
            on_frame=lambda frame, stats: None,  # sẽ được ghi đè bởi UI
            camera_index=0,
            target_fps=20,
        )
 
    # 3. Khởi động Tkinter UI
    root = tk.Tk()
    root.geometry("1060x600")
    app = MaskDetectionUI(root, camera_factory)
 
    print("[main] Giao diện đã khởi động.")
    root.mainloop()
 
 
if __name__ == "__main__":
    main()