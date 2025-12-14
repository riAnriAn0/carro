from camera.camera import CameraThread
from inferencia.yolo_inference import YoloInference
from comm.udp_sender import UDPSender
import cv2
import time

CLASSES = ['faixa-central']

def main():

    camera = CameraThread().start()

    yolo = YoloInference()
    
    udp_sender = UDPSender()
    udp_sender.start()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            detections = yolo.predict(frame)

            boxes = detections["boxes"]
            scores = detections["scores"]
            classes = detections["classes"]

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,f"{CLASSES[cls]} {score:.2f}",(x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1)

                print(f"Classe: {cls} | "f"Score: {score:.2f} | "f"Box: ({x1},{y1})-({x2},{y2})")

            if yolo.infer_fps is not None:
                cv2.putText(frame,f"Infer FPS: {yolo.infer_fps:.1f}",(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 255),2)

            udp_sender.update_frame(frame)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        udp_sender.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()