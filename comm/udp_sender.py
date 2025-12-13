import socket
import numpy as np
import cv2
import threading
from time import sleep

class UDPSender:
    def __init__(self, ip, port, jpeg_quality):
        self.ip = ip
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        self.frame = None
        self.thread = None

    def start(self):
        if self.thread is not None:
            return

        self.running = True
        self.thread = threading.Thread(
            target=self.send_frame,
            daemon=True
        )
        self.thread.start()

    def update_frame(self, frame):
        self.frame = frame

    def send_frame(self):
        while self.running:
            if self.frame is None:
                sleep(0.01)
                continue

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            ret, jpg = cv2.imencode('.jpg', self.frame, encode_param)
            if ret:
                self.sock.sendto(jpg.tobytes(), (self.ip, self.port))

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.5)
        self.sock.close()
