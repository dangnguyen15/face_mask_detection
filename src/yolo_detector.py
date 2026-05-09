from ultralytics import YOLO

model = YOLO("models/yolov3.pt")

def detect_faces(frame):

    results = model(frame)

    faces = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            w = x2 - x1
            h = y2 - y1

            faces.append((x1, y1, w, h))

    return faces