# ocr_engine_b/onnx_accelerator.py
import os
from paddleocr import PaddleOCR


class ONNXOCRAccelerator:
    """
    ONNX Runtime 推理加速器。
    加载 INT8/FP32 量化后的 ONNX 模型，替换 PaddleOCR 默认的 Paddle 推理后端。
    """

    def __init__(self, precision="INT8", unclip_ratio=1.5):
        """
        :param precision: 模型精度，"INT8" 或 "FP32"
        :param unclip_ratio: 修复[致命-2]：DBNet文本框膨胀系数，传入det_db_unclip_ratio
        """
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

        def get_model_path(model_type):
            """拼接模型路径并校验文件是否存在，不存在时给出明确提示。"""
            path = os.path.join(base, f"{model_type}_onnx_{precision.lower()}", "model.onnx")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"模型文件不存在: {path}\n"
                    f"请先运行 model_quantization.py 完成 {precision} 量化导出。"
                )
            return path

        self.engine = PaddleOCR(
            det_model_dir=get_model_path("det"),
            rec_model_dir=get_model_path("rec"),
            cls_model_dir=get_model_path("cls"),
            use_angle_cls=True,
            use_gpu=False,
            use_onnx=True,
            show_log=False,
            det_db_unclip_ratio=unclip_ratio,  # 修复[致命-2]：实际传入检测器的膨胀系数
        )
