# ocr_engine_b/paddle_ocr_engine.py
import cv2
import time
import threading
import numpy as np
from paddleocr import PaddleOCR

from common.utils import load_config, check_image_validity
from common.exception_handle import ImageReadError, OCRInferenceError
from preprocess.preprocess_switch import run_preprocess
from post_process.box_merging import merge_boxes
from post_process.semantic_correction import SemanticCorrector


class AlgorithmB:
    """
    方案B核心调度类：基于PaddleOCR（DBNet检测 + CRNN识别）的深度学习OCR引擎。
    支持通过ONNX Runtime加载INT8量化模型进行推理加速。
    """

    def __init__(self):
        self.ocr = None
        self.lock = threading.Lock()
        self._current_unclip_ratio = None
        self.corrector = SemanticCorrector(mode="dict")

    def load_model(self, unclip_ratio=1.5):
        """
        延迟加载OCR模型。若模型已加载且unclip_ratio未变，则复用已有实例；
        否则重新初始化以应用新的膨胀系数。
        """
        with self.lock:
            if self.ocr is not None and self._current_unclip_ratio == unclip_ratio:
                return
            cfg = load_config().get("ocr_b", {})
            if cfg.get("inference_engine") == "onnx":
                from .onnx_accelerator import ONNXOCRAccelerator
                self.ocr = ONNXOCRAccelerator(
                    cfg.get("model_precision", "INT8"),
                    unclip_ratio=unclip_ratio,
                ).engine
            else:
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    use_gpu=False,
                    show_log=False,
                    det_db_unclip_ratio=unclip_ratio,
                )
            self._current_unclip_ratio = unclip_ratio

    def detect_and_recognize(self, path, conf_threshold=0.5, **kwargs):
        """
        执行完整的检测+识别+后处理管线。
        :param path: 图像文件路径
        :param conf_threshold: 置信度过滤阈值
        :param unclip_ratio: 文本框膨胀系数（对应UI"框膨胀率"滑块）
        :return: (识别文本, 可视化图像, 耗时秒数)
        :raises ImageReadError: 图像文件无法读取时抛出
        :raises OCRInferenceError: PaddleOCR推理失败时抛出
        """
        unclip_ratio = kwargs.get("unclip_ratio", 1.5)
        self.load_model(unclip_ratio=unclip_ratio)

        start = time.time()
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 统一校验：None → ImageReadError，size==0 → ImageEmptyError
        check_image_validity(img, path)

        # 预处理后得到BGR三通道图像
        preprocessed = run_preprocess(img, scheme="B")

        try:
            res = self.ocr.ocr(preprocessed, cls=True)
        except Exception as e:
            raise OCRInferenceError(f"PaddleOCR推理失败: {e}") from e

        # 解析结果，过滤低置信度检测框
        structured = []
        if res and res[0]:
            for line in res[0]:
                if line[1][1] >= conf_threshold:
                    # DBNet 返回四边形顶点（顺时针：左上→右上→右下→左下）
                    # 取各顶点 x/y 的 min/max，兼容旋转文本框，避免直接取 [0][2] 导致坐标偏差
                    pts = line[0]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    structured.append({
                        "box": [min(xs), min(ys), max(xs), max(ys)],
                        "text": line[1][0],
                        "conf": line[1][1],
                    })

        # 后处理：碎框合并
        merged = merge_boxes(structured)

        # 在原图上绘制检测框
        for m in merged:
            cv2.rectangle(
                img,
                (int(m["box"][0]), int(m["box"][1])),
                (int(m["box"][2]), int(m["box"][3])),
                (0, 255, 0), 2,
            )

        # 对每行识别结果应用语义纠错
        corrected_texts = [self.corrector.correct(m["text"]) for m in merged]
        return "\n".join(corrected_texts), img, time.time() - start
