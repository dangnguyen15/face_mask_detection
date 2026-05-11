import cv2
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

cascade_path = os.path.join(
    BASE_DIR,
    "assets",
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)


def detect_faces(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )

    results = []

    for (x, y, w, h) in faces:

        results.append((x, y, x + w, y + h))

    return results