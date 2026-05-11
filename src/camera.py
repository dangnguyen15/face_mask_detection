import cv2

class Camera:
    def __init__(self):
        # Dùng CAP_DSHOW để load camera nhanh hơn trên Windows
        self.vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            print("Lỗi: Không thể mở webcam!")

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, frame
        return False, None

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()