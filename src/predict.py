import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

class MaskPredictor:
    def __init__(self, model_path):
        # 1. Nạp mô hình CNN của bạn
        print("[INFO] Đang nạp model phân loại khẩu trang...")
        self.model = load_model(model_path)
        self.img_size = 128
        
        # 2. Nạp bộ nhận diện khuôn mặt DNN (Stage 1)
        # Đảm bảo tên file khớp chính xác với file bạn đã tải
        prototxt = "deploy.prototxt" 
        weights = "res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists(prototxt) or not os.path.exists(weights):
            print(f"[ERROR] Không tìm thấy file {prototxt} hoặc {weights}!")
            print("Vui lòng kiểm tra lại tên file trong thư mục gốc.")
            
        self.face_net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        print("[INFO] Đã nạp xong bộ nhận diện khuôn mặt DNN.")

    def predict(self, frame):
        if frame is None: return []
        
        h, w = frame.shape[:2]
        # Tiền xử lý ảnh cho DNN (Stage 1)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        results = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Chỉ lấy các vùng có độ tin cậy là mặt người > 50%
            if confidence > 0.5:
                # Tính toán tọa độ khung bao
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Đảm bảo khung bao nằm trong kích thước ảnh
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                
                # Cắt vùng mặt để quét khẩu trang (Stage 2)
                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size == 0: continue

                # Tiền xử lý vùng mặt cho model CNN của bạn
                face_input = cv2.resize(face_roi, (self.img_size, self.img_size))
                face_input = face_input / 255.0
                face_input = np.expand_dims(face_input, axis=0)

                # Dự đoán khẩu trang
                prediction = self.model.predict(face_input, verbose=0)[0][0]
                
                # Phân loại nhãn (Threshold 0.5)
                # 0 thường là Mask, 1 thường là No Mask (kiểm tra lại dataset của bạn)
                label = "No Mask" if prediction < 0.5 else "Mask"
                color = (0, 0, 225) if label == "No Mask" else (0, 225, 0)
                
                # Tính toán độ tin cậy hiển thị (Accuracy)
                acc = (1 - prediction) * 100 if label == "Mask" else prediction * 100
                
                results.append({
                    "box": (startX, startY, endX - startX, endY - startY),
                    "label": f"{label} ({acc:.1f}%)",
                    "color": color
                })
        return results