# ocr_engine_a/tesseract_engine.py
import cv2, time, numpy as np
from preprocess.preprocess_switch import run_preprocess
from .recognition.tesseract_call import dynamic_psm_recognition
class AlgorithmA:
    def detect_and_recognize(self, path, conf_threshold=0.5, **kwargs):
        start = time.time()
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        dilated, binary_n = run_preprocess(img, scheme="A", **kwargs)
        boxes = sorted([cv2.boundingRect(c) for c in cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]], key=lambda b: b[1])
        txts = []
        valid_box_count = 0  # 🌟 新增：有效调用计数器

        for x, y, w, h in boxes:
            if w > 20 and h > 20:
                valid_box_count += 1  # 🌟 新增：每次进入识别前 +1
                txt = dynamic_psm_recognition(
                    cv2.copyMakeBorder(binary_n[y:y + h, x:x + w], 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                                       value=[255, 255, 255]), conf_threshold)
                if txt: txts.append(txt); cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 🌟 新增：在控制台打印出真实的 I/O 调用次数
        print(f"\n[性能探针] 提取出 {len(boxes)} 个原始轮廓，实际密集唤醒 Tesseract: {valid_box_count} 次")

        return "\n".join(txts), img, time.time() - start

    def get_morphology_preview(self, path, erode_size, dilate_x):
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            # 仅执行预处理，不执行 OCR，以保证实时滑动的流畅度
            dilated, _ = run_preprocess(img, scheme="A", erode_size=erode_size, dilate_x=dilate_x, dilate_y=3)
            return dilated
        except Exception:
            return None
