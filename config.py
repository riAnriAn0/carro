#----------- TRANSMISSÃO UDP
UDP_IP_PC = "192.168.3.58"
UDP_IP_ENVIO = "0.0.0.0"
UDP_PORT = 9999
JPEG_QUALITY = 80


# ---------- MODELO
MODEL_PATH = "inferencia/modelo/yolo-tf.tflite"
CLASSES = ['faixa-central']

# ---------- CAMERA
CAM_ID = 0
CAP_WIDTH, CAP_HEIGHT = 160, 160
NUM_THREADS = 4


# ---------- INFERÊNCIA
SCORE_TH = 0.5
NMS_IOU = 0.45
TOP_K = 200