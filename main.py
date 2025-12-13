from camera.camera import CameraThread
from inferencia.yolo_inference import YoloInference
from comm.udp_sender import UDPSender
import config


def main():
    # Inicia a câmera
    camera = CameraThread(
        cam_id=config.CAM_ID,
        width=config.CAP_WIDTH,
        height=config.CAP_HEIGHT
    ).start()

    # Inicia o modelo de inferência
    yolo = YoloInference(
        model_path=config.MODEL_PATH,
        num_threads=config.NUM_THREADS,
        score_th=config.SCORE_TH,
        nms_iou=config.NMS_IOU,
        top_k=config.TOP_K
    )

    # Inicia o transmissor UDP
    udp_sender = UDPSender(
        ip=config.UDP_IP_PC,
        port=config.UDP_PORT,
        jpeg_quality=config.JPEG_QUALITY
    )
    udp_sender.start()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            # Preprocessa o frame
            yolo.preprocess(frame)

            # Atualiza o frame a ser enviado via UDP
            udp_sender.update_frame(frame)

    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        udp_sender.stop()   

if __name__ == "__main__":
    main()
