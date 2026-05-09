import cv2


def draw_label(frame, text, confidence):

    if text == "Mask":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    label = f"{text} : {confidence:.2f}"

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    return frame