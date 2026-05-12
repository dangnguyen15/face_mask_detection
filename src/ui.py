import tkinter as tk
from PIL import Image, ImageTk
import cv2
 
 
class MaskApp:
    def __init__(self, root, camera, predictor):
        self.root = root
        self.root.title("Face Mask Detection")
        self.camera = camera
        self.predictor = predictor
 
        # ── Window styling ──────────────────────────────────
        self.root.configure(bg="#0a0c0f")
        self.root.resizable(False, False)
 
        # ── Header bar ──────────────────────────────────────
        header = tk.Frame(self.root, bg="#11151a", height=48)
        header.pack(fill="x")
        header.pack_propagate(False)
 
        tk.Label(
            header,
            text="◈  HSU — FACE MASK DETECTION SYSTEM",
            bg="#11151a", fg="#00e5ff",
            font=("Courier New", 11, "bold")
        ).pack(side="left", padx=16, pady=12)
 
        tk.Frame(self.root, bg="#00e5ff", height=1).pack(fill="x")
 
        # ── Canvas wrapper with border ───────────────────────
        canvas_wrap = tk.Frame(self.root, bg="#1e2832", padx=2, pady=2)
        canvas_wrap.pack(padx=16, pady=12)
 
        self.canvas = tk.Canvas(canvas_wrap, width=640, height=480,
                                bg="#050709", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack()
 
        # HUD corner marks drawn once
        self._draw_corner_marks()
 
        # ── Footer bar ───────────────────────────────────────
        footer = tk.Frame(self.root, bg="#11151a", height=28)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(
            footer,
            text="CAM-01  ·  LIVE  ·  HSU v2.0",
            bg="#11151a", fg="#3a4a5a",
            font=("Courier New", 8)
        ).pack(side="right", padx=12)
 
        self.update()
 
    def _draw_corner_marks(self):
        """HUD-style corner markers on the canvas."""
        c, L, T = self.canvas, 18, 2
        col = "#00e5ff"
        c.create_line(T,       T,       T+L,     T,       fill=col, width=1)
        c.create_line(T,       T,       T,       T+L,     fill=col, width=1)
        c.create_line(638-T,   T,     638-T-L,   T,       fill=col, width=1)
        c.create_line(638-T,   T,     638-T,     T+L,     fill=col, width=1)
        c.create_line(T,     478-T,    T+L,     478-T,    fill=col, width=1)
        c.create_line(T,     478-T,    T,       478-T-L,  fill=col, width=1)
        c.create_line(638-T, 478-T,  638-T-L,  478-T,    fill=col, width=1)
        c.create_line(638-T, 478-T,  638-T,    478-T-L,  fill=col, width=1)
 
    def update(self):
        ret, frame = self.camera.get_frame()
        if ret:
            # Gọi predictor để lấy danh sách khuôn mặt
            results = self.predictor.predict(frame)
 
            # QUAN TRỌNG: Phải có vòng lặp này để vẽ lên frame trước khi hiển thị
            for res in results:
                x, y, w, h = res["box"]
 
                # Thanh label mờ phía trên box
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y - 22), (x + w, y), res["color"], -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
 
                # Bounding box chính
                cv2.rectangle(frame, (x, y), (x+w, y+h), res["color"], 2)
 
                # Corner accents trên box
                clen = 10
                for px, py, dx, dy in [
                    (x,   y,    1,  1), (x+w, y,   -1,  1),
                    (x,   y+h,  1, -1), (x+w, y+h, -1, -1),
                ]:
                    cv2.line(frame, (px, py), (px + dx*clen, py), (255,255,255), 2)
                    cv2.line(frame, (px, py), (px, py + dy*clen), (255,255,255), 2)
 
                cv2.putText(frame, res["label"].upper(), (x + 4, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                            cv2.LINE_AA)
 
            # Hiển thị frame đã vẽ lên Canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self._draw_corner_marks()   # vẽ lại HUD sau mỗi frame
 
        self.root.after(15, self.update)