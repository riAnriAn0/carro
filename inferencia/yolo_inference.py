import time
import numpy as np
import tflite_runtime.interpreter as tflite

MODEL_PATH = "./modelo/yolo-tf.tflite" # modelo integer quantizado
SCORE_TH = 0.5
NMS_IOU = 0.45
TOP_K = 200



class YoloInference:
    def __init__(
        self,
        model_path = MODEL_PATH,
        num_threads=2,
        score_th=0.25,
        iou_th=0.45,
        top_k=200,
    ):
        self.score_th = score_th
        self.iou_th = iou_th
        self.top_k = top_k

        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.in_det = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]

        _, self.h_model, self.w_model, _ = self.in_det["shape"]
        self.dtype = self.in_det["dtype"]

        self.prev_ts = None
        self.infer_fps = None
        self.alpha = 0.18

        self.input_data = np.zeros(
            (1, self.h_model, self.w_model, 3),
            dtype=np.float32
        )

    # -------------------------------------------------
    # Inferência principal
    # -------------------------------------------------
    def predict(self, frame):
        t0 = time.perf_counter()

        self._preprocess(frame)
        self.interpreter.set_tensor(self.in_det["index"], self.input_data)
        self.interpreter.invoke()

        raw = self.interpreter.get_tensor(self.out_det["index"])

        detections = self._postprocess(raw, frame.shape[:2])

        self._update_fps(t0)

        detections["infer_fps"] = self.infer_fps
        return detections

    # -------------------------------------------------
    # Pré-processamento
    # -------------------------------------------------
    def _preprocess(self, frame):
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.w_model, self.h_model))

        if self.dtype == np.float32:
            self.input_data[0] = resized.astype(np.float32) / 255.0
        else:
            self.input_data[0] = resized.astype(np.uint8)

    # -------------------------------------------------
    # Pós-processamento
    # -------------------------------------------------
    def _postprocess(self, raw_out, frame_shape):
        fh, fw = frame_shape
        arr = np.squeeze(raw_out).T  # [N, 84]

        boxes_raw = arr[:, :4]
        class_scores = arr[:, 4:]

        if np.max(class_scores) > 1.5 or np.min(class_scores) < -0.5:
            probs = 1.0 / (1.0 + np.exp(-class_scores))
        else:
            probs = class_scores

        classes = np.argmax(probs, axis=1)
        scores = probs[np.arange(len(classes)), classes]

        mask = scores > self.score_th
        if not np.any(mask):
            return self._empty_result()

        idxs = np.where(mask)[0]
        if idxs.size > self.top_k:
            idxs = idxs[np.argsort(scores[idxs])[-self.top_k:]]

        sel_boxes = boxes_raw[idxs]
        sel_scores = scores[idxs]
        sel_classes = classes[idxs]

        # cx,cy,w,h → x1,y1,x2,y2
        cx, cy, bw, bh = sel_boxes.T
        x1 = (cx - bw / 2) * self.w_model
        y1 = (cy - bh / 2) * self.h_model
        x2 = (cx + bw / 2) * self.w_model
        y2 = (cy + bh / 2) * self.h_model

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # escala para frame real
        boxes[:, [0, 2]] *= fw / self.w_model
        boxes[:, [1, 3]] *= fh / self.h_model

        keep = self._nms(boxes, sel_scores)

        return {
            "boxes": boxes[keep],
            "scores": sel_scores[keep],
            "classes": sel_classes[keep],
        }

    # -------------------------------------------------
    # NMS
    # -------------------------------------------------
    def _nms(self, boxes, scores):
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-9)
            order = order[np.where(iou <= self.iou_th)[0] + 1]

        return keep

    # -------------------------------------------------
    # FPS por inferência
    # -------------------------------------------------
    def _update_fps(self, t0):
        dt = time.perf_counter() - t0
        fps = 1.0 / dt if dt > 0 else 0.0

        if self.infer_fps is None:
            self.infer_fps = fps
        else:
            self.infer_fps = (1 - self.alpha) * self.infer_fps + self.alpha * fps

    def _empty_result(self):
        return {
            "boxes": np.empty((0, 4)),
            "scores": np.empty((0,)),
            "classes": np.empty((0,), dtype=int),
        }
