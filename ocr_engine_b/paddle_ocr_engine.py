# ocr_engine_b/paddle_ocr_engine.py
import cv2, time, numpy as np, threading
from paddleocr import PaddleOCR
from common.utils import load_config
from preprocess.preprocess_switch import run_preprocess
from post_process.box_merging import merge_boxes
class AlgorithmB:
    def __init__(self): self.ocr = None; self.lock = threading.Lock()
    def load_model(self):
        with self.lock:
            if self.ocr: return
            cfg = load_config().get("ocr_b", {})
            if cfg.get("inference_engine") == "onnx":
                from .onnx_accelerator import ONNXOCRAccelerator
                self.ocr = ONNXOCRAccelerator(cfg.get("model_precision", "INT8")).engine
            else: self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, show_log=False)
    def detect_and_recognize(self, path, conf_threshold=0.5, **kwargs):
        self.load_model()
        start = time.time()
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        res = self.ocr.ocr(run_preprocess(img, scheme="B"), cls=True)
        structured = []
        if res and res[0]:
            for line in res[0]:
                if line[1][1] >= conf_threshold: structured.append({"box": [line[0][0][0], line[0][0][1], line[0][2][0], line[0][2][1]], "text": line[1][0], "conf": line[1][1]})
        merged = merge_boxes(structured)
        for m in merged: cv2.rectangle(img, (int(m["box"][0]), int(m["box"][1])), (int(m["box"][2]), int(m["box"][3])), (0, 255, 0), 2)
        return "\n".join([m["text"] for m in merged]), img, time.time() - start
