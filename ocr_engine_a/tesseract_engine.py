# ocr_engine_a/tesseract_engine.py
import cv2
import time
import numpy as np

from preprocess.preprocess_switch import run_preprocess
from .recognition.tesseract_call import dynamic_psm_recognition
from post_process.semantic_correction import SemanticCorrector
from common.utils import load_config, check_image_validity


class AlgorithmA:
    """
    方案A核心调度类：基于OpenCV形态学轮廓检测 + Tesseract识别的传统机器视觉OCR引擎。
    """

    def __init__(self):
        self.corrector = SemanticCorrector(mode="dict")

    def detect_and_recognize(self, path, conf_threshold=0.5, **kwargs):
        """
        执行形态学检测 + Tesseract识别 + 字符过滤后处理完整管线。
        :param path: 图像文件路径
        :param conf_threshold: Tesseract置信度过滤阈值
        :return: (识别文本, 可视化图像, 耗时秒数)
        :raises ImageReadError: 图像文件无法读取时抛出
        """
        start = time.time()
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 统一校验：None → ImageReadError，size==0 → ImageEmptyError
        check_image_validity(img, path)

        dilated, binary_n = run_preprocess(img, scheme="A", **kwargs)

        # 提取外部轮廓并按Y轴排序，模拟自然阅读顺序
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[1])

        # 配置加载提升到循环外，避免每个 ROI 都重复读磁盘
        ocr_config = load_config().get("ocr_a", {})

        txts = []
        for x, y, w, h in boxes:
            if w > 20 and h > 20:
                roi = cv2.copyMakeBorder(
                    binary_n[y:y + h, x:x + w],
                    10, 10, 10, 10,
                    cv2.BORDER_CONSTANT, value=[255, 255, 255],
                )
                txt = dynamic_psm_recognition(roi, conf_threshold, config=ocr_config)
                if txt:
                    corrected = self.corrector.correct(txt)
                    txts.append(corrected)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return "\n".join(txts), img, time.time() - start

    def get_morphology_preview(self, path, erode_size, dilate_x):
        """
        仅执行预处理，不执行OCR，用于UI实时预览形态学参数效果。
        :return: 膨胀处理后的二值图，或None（读取失败时）
        """
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            dilated, _ = run_preprocess(img, scheme="A", erode_size=erode_size, dilate_x=dilate_x, dilate_y=3)
            return dilated
        except Exception:
            return None
