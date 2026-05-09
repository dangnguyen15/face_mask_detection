import cv2

from PIL import Image
from PIL import ImageTk

from src.predict import predict_mask
from src.yolo_detector import detect_faces


def run_camera(video_label, root):

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    def update_frame():

        ret, frame = cap.read()

        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # =========================
        # YOLO FACE DETECTION
        # =========================

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:

            # Crop face

            face = frame[y:y+h, x:x+w]

            try:

                result, confidence = predict_mask(face)

                # Color

                if result == "Mask":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                # Bounding box

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    color,
                    2
                )

                # Text

                text = f"{result} ({confidence*100:.1f}%)"

                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            except Exception as e:

                print("Prediction Error:", e)

        # =========================
        # DISPLAY
        # =========================

        frame = cv2.resize(frame, (900, 550))

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        img = Image.fromarray(frame_rgb)

        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk

        video_label.configure(image=imgtk)

        root.after(10, update_frame)

    update_frame()