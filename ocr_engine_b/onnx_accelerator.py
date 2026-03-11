# ocr_engine_b/onnx_accelerator.py
import os, onnxruntime as ort
from paddleocr import PaddleOCR
class ONNXOCRAccelerator:
    def __init__(self, precision="INT8"):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        def get_p(m): return os.path.join(base, f"{m}_onnx_{precision.lower()}", "model.onnx")
        self.engine = PaddleOCR(det_model_dir=get_p("det"), rec_model_dir=get_p("rec"), cls_model_dir=get_p("cls"), use_angle_cls=True, use_gpu=False, use_onnx=True, show_log=False)
