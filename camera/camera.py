import cv2
import threading
from time import sleep

CAM_ID = "./video/video.mp4"
CAP_WIDTH, CAP_HEIGHT = 160, 160
NUM_THREADS = 4

# ===== Camera thread =====
class CameraThread:
    def __init__(self, cam_id=CAM_ID, width=CAP_WIDTH, height=CAP_HEIGHT):
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # type: ignore
        self.lock = threading.Lock()
        self.frame = None
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def _reader(self):
        while self.running:
            ret, f = self.cap.read()
            if not ret:
                sleep(0.01)
                continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=0.5)
        self.cap.release()
