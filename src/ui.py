import tkinter as tk
from PIL import Image, ImageTk
import cv2

class MaskApp:
    def __init__(self, root, camera, predictor):
        self.root = root
        self.root.title("HSU - Face Mask Detection System")
        self.camera = camera
        self.predictor = predictor
        
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()
        
        self.update()

    def update(self):
        ret, frame = self.camera.get_frame()
        if ret:
            # Gọi predictor để lấy danh sách khuôn mặt
            results = self.predictor.predict(frame)
        
            # QUAN TRỌNG: Phải có vòng lặp này để vẽ lên frame trước khi hiển thị
            for res in results:
                x, y, w, h = res["box"]
                cv2.rectangle(frame, (x, y), (x+w, y+h), res["color"], 2)
                cv2.putText(frame, res["label"], (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, res["color"], 2)

            # Hiển thị frame đã vẽ lên Canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    
        self.root.after(15, self.update)