import os
# Giảm bớt log thừa của TensorFlow và ép chạy CPU để tránh treo GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tkinter as tk
from src.camera import Camera
from src.predict import MaskPredictor
from src.ui import MaskApp

def main():
    print("[INFO] Đang khởi động hệ thống...")
    
    MODEL_PATH = "models/face_mask_model.keras"
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Không tìm thấy file model tại: {MODEL_PATH}")
        return

    print("[INFO] 1/3: Đang nạp mô hình AI...")
    predictor = MaskPredictor(MODEL_PATH)
    
    print("[INFO] 2/3: Đang kết nối Camera...")
    camera = Camera()
    
    print("[INFO] 3/3: Đang mở giao diện người dùng...")
    root = tk.Tk()
    app = MaskApp(root, camera, predictor)
    
    print("[SUCCESS] Hệ thống đã sẵn sàng!")
    root.mainloop()

if __name__ == "__main__":
    main()