import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
from time import perf_counter


class YoloInference:
    def __init__(self, model_path, num_threads=2,score_th=0.25, nms_iou=0.45, top_k=200):

        self.interpreter = tflite.Interpreter(model_path=model_path,num_threads=num_threads)
        self.interpreter.allocate_tensors()

        self.in_det = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]

        _, self.h_model, self.w_model, _ = self.in_det['shape']
        self.dtype = self.in_det['dtype']

        self.score_th = score_th
        self.nms_iou = nms_iou
        self.top_k = top_k

        self.prev_infer_ts = None
        self.smoothed_infer_fps = None
        self.alpha = 0.18

        self.input_data = np.zeros((1, self.h_model, self.w_model, 3),dtype=np.float32)

    # ---------- PREPROCESS ----------
    def preprocess(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.w_model, self.h_model))

        if self.dtype == np.float32:
            self.input_data[0] = resized.astype(np.float32) / 255.0
        else:
            self.input_data[0] = resized.astype(np.uint8)

    # ---------- NMS ----------
    @staticmethod
    def nms_numpy(boxes, scores, iou_threshold):
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

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)

            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

            order = order[np.where(iou <= iou_threshold)[0] + 1]

        return keep

    # ---------- INFER ----------
    def infer(self):
        t0 = perf_counter()

        self.interpreter.set_tensor(self.in_det['index'], self.input_data)
        self.interpreter.invoke()

        t1 = perf_counter()

        # FPS real de inferência
        if self.prev_infer_ts is not None:
            dt = t1 - self.prev_infer_ts
            fps = 1.0 / dt if dt > 0 else 0.0

            if self.smoothed_infer_fps is None:
                self.smoothed_infer_fps = fps
            else:
                self.smoothed_infer_fps = (
                    (1 - self.alpha) * self.smoothed_infer_fps +
                    self.alpha * fps
                )

        self.prev_infer_ts = t1

        return self.interpreter.get_tensor(self.out_det['index'])

    # ---------- POSTPROCESS ----------
    def postprocess(self, raw_out, frame_shape):
        arr = np.squeeze(raw_out).T

        boxes_raw = arr[:, :4]
        class_scores = arr[:, 4:]

        if np.max(class_scores) > 1.5 or np.min(class_scores) < -0.5:
            class_probs = 1 / (1 + np.exp(-class_scores))
        else:
            class_probs = class_scores

        best_class = np.argmax(class_probs, axis=1)
        best_scores = class_probs[np.arange(len(best_class)), best_class]

        mask = best_scores > self.score_th
        if not np.any(mask):
            return [], []

        idxs = np.where(mask)[0]
        idxs = idxs[np.argsort(best_scores[idxs])[-self.top_k:]]

        sel_boxes = boxes_raw[idxs]
        sel_scores = best_scores[idxs]

        cx, cy, bw, bh = sel_boxes.T
        x1 = (cx - bw / 2) * self.w_model
        y1 = (cy - bh / 2) * self.h_model
        x2 = (cx + bw / 2) * self.w_model
        y2 = (cy + bh / 2) * self.h_model

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        fh, fw = frame_shape
        boxes[:, [0, 2]] *= fw / self.w_model
        boxes[:, [1, 3]] *= fh / self.h_model

        keep = self.nms_numpy(boxes, sel_scores, self.nms_iou)

        return boxes[keep], sel_scores[keep]

    # ---------- API FINAL ----------
    def predict(self, frame):
        self.preprocess(frame)
        raw_out = self.infer()
        boxes, scores = self.postprocess(raw_out, frame.shape[:2])
        return boxes, scores, self.smoothed_infer_fps
