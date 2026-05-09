import cv2
import numpy as np

from tensorflow.keras.models import load_model


IMG_SIZE = 224

MODEL_PATH = "models/face_mask_model.keras"


model = load_model(MODEL_PATH)

labels = [
    "Mask",
    "No Mask"
]


def predict_mask(face):

    face = cv2.resize(
        face,
        (IMG_SIZE, IMG_SIZE)
    )

    face = face / 255.0

    face = np.expand_dims(face, axis=0)

    prediction = model.predict(
        face,
        verbose=0
    )

    class_index = np.argmax(prediction)

    confidence = prediction[0][class_index]

    return labels[class_index], confidence