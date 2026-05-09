import tkinter as tk
from PIL import Image, ImageTk
import cv2

from src.predict import predict_mask


class FaceMaskUI:

    def __init__(self, root):

        self.root = root

        self.root.title("Face Mask Detection")

        self.root.geometry("1000x700")

        self.root.configure(bg="#1e1e1e")

        self.cap = None

        self.running = False

        # =========================
        # TITLE
        # =========================

        title = tk.Label(
            root,
            text="FACE MASK DETECTION",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#1e1e1e"
        )

        title.pack(pady=20)

        # =========================
        # VIDEO LABEL
        # =========================

        self.video_label = tk.Label(root)

        self.video_label.pack()

        # =========================
        # BUTTONS
        # =========================

        btn_frame = tk.Frame(root, bg="#1e1e1e")

        btn_frame.pack(pady=20)

        start_btn = tk.Button(
            btn_frame,
            text="Start Detection",
            font=("Arial", 14, "bold"),
            bg="green",
            fg="white",
            width=20,
            height=2,
            command=self.start_camera
        )

        start_btn.grid(row=0, column=0, padx=20)

        stop_btn = tk.Button(
            btn_frame,
            text="Stop",
            font=("Arial", 14, "bold"),
            bg="orange",
            fg="white",
            width=20,
            height=2,
            command=self.stop_camera
        )

        stop_btn.grid(row=0, column=1, padx=20)

        exit_btn = tk.Button(
            root,
            text="Exit",
            font=("Arial", 14, "bold"),
            bg="red",
            fg="white",
            width=20,
            height=2,
            command=self.close_app
        )

        exit_btn.pack(pady=10)

    # =========================
    # START CAMERA
    # =========================

    def start_camera(self):

        if self.running:
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print("Cannot open webcam")
            return

        self.running = True

        self.update_frame()

    # =========================
    # UPDATE FRAME
    # =========================

    def update_frame(self):

        if not self.running:
            return

        ret, frame = self.cap.read()

        if ret:

            frame = cv2.flip(frame, 1)

            # Resize frame

            frame = cv2.resize(frame, (800, 500))

            try:

                result, confidence = predict_mask(frame)

                if result == "Mask":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    f"{result} ({confidence:.2f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

            except Exception as e:

                print("Prediction Error:", e)

            # BGR -> RGB

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            img = Image.fromarray(frame_rgb)

            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk

            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    # =========================
    # STOP CAMERA
    # =========================

    def stop_camera(self):

        self.running = False

        if self.cap:
            self.cap.release()

    # =========================
    # CLOSE APP
    # =========================

    def close_app(self):

        self.stop_camera()

        self.root.destroy()


# =========================
# RUN UI
# =========================

def run_ui():

    root = tk.Tk()

    app = FaceMaskUI(root)

    root.mainloop()

