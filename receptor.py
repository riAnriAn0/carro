import socket
import numpy as np
import cv2

UDP_IP = "0.0.0.0"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(65535)
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is not None:
        cv2.imshow("Stream UDP", img)

    if cv2.waitKey(1) == 27:
        break

sock.close()
cv2.destroyAllWindows()
