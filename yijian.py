import os
import json

# ==========================================
# 核心配置文件内容 (修正了 True/False 语法)
# ==========================================
PARAMS_CONFIG = {
    "preprocess": {
        "preprocess_mode": "scheme_b",
        "clahe_clip_limit": 2.0,
        "angle_fix_threshold": 15,
        "morph_erode_size": 2,
        "morph_dilate_x": 15,
        "morph_dilate_y": 3
    },
    "ocr_a": {
        "tesseract_cmd_path": "D:\\Tesseract-OCR\\tesseract.exe",
        "lang": "chi_sim+eng"
    },
    "ocr_b": {
        "use_angle_cls": True,  # 修正：小写 true 改为大写 True
        "lang": "ch",
        "use_gpu": False,  # 修正：小写 false 改为大写 False
        "model_precision": "INT8",
        "inference_engine": "onnx"
    },
    "box_merging": {
        "horizontal_threshold": 15,
        "height_threshold": 0.1
    },
    "semantic_correction": {
        "correction_mode": "dict",
        "llm_model_path": "./models/qwen-0.5b"
    }
}

# ==========================================
# 全量代码映射表 (其余逻辑保持不变)
# ==========================================
PROJECT_FILES = {
    "common/exception_handle.py": r'''# common/exception_handle.py
class OCRBaseError(Exception): pass
class ImageReadError(OCRBaseError): pass
class ImageEmptyError(OCRBaseError): pass
class ParameterError(OCRBaseError): pass
class OCRInferenceError(OCRBaseError): pass
''',

    "common/utils.py": r'''# common/utils.py
import json, os
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "params.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}
def check_image_validity(image, path=""):
    if image is None or image.size == 0: raise Exception(f"图像无效: {path}")
''',

    "preprocess/clahe_enhance.py": r'''# preprocess/clahe_enhance.py
import cv2
def apply_clahe(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if clip_limit <= 0: clip_limit = 2.0
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(gray_img)
''',

    "preprocess/orientation_fix.py": r'''# preprocess/orientation_fix.py
import cv2, numpy as np
def fix_orientation(image, max_angle_threshold=15.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0: return image, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    angle = angle - 90.0 if angle >= 45.0 else (angle + 90.0 if angle < -45.0 else angle)
    if abs(angle) < 0.5: return image, 0.0
    if abs(angle) > max_angle_threshold: return image, angle
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)), angle
''',

    "preprocess/preprocess_switch.py": r'''# preprocess/preprocess_switch.py
import cv2
from .clahe_enhance import apply_clahe
from .orientation_fix import fix_orientation
from common.utils import load_config
def run_preprocess(image, scheme="B", **kwargs):
    config = load_config().get("preprocess", {})
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    enhanced = apply_clahe(gray, clip_limit=config.get("clahe_clip_limit", 2.0))
    fixed_img, _ = fix_orientation(enhanced, max_angle_threshold=config.get("angle_fix_threshold", 15))
    if scheme == "B": return fixed_img
    binary = cv2.adaptiveThreshold(cv2.medianBlur(fixed_img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    eroded = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (int(kwargs.get("erode_size", 2)), int(kwargs.get("erode_size", 2)))))
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (int(kwargs.get("dilate_x", 15)), int(kwargs.get("dilate_y", 3)))))
    return dilated, cv2.bitwise_not(binary)
''',

    "ocr_engine_a/recognition/tesseract_call.py": r'''# ocr_engine_a/recognition/tesseract_call.py
import pytesseract, re
from common.utils import load_config
def dynamic_psm_recognition(roi, conf_threshold):
    config = load_config().get("ocr_a", {})
    pytesseract.pytesseract.tesseract_cmd = config.get("tesseract_cmd_path")
    h, w = roi.shape[:2]
    psm = 10 if h>0 and 0.8<w/h<1.2 else (3 if h>50 and w/h<2.5 else 7)
    data = pytesseract.image_to_data(roi, lang=config.get('lang'), config=f'--oem 3 --psm {psm}', output_type=pytesseract.Output.DICT)
    text = ""
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word and str(data['conf'][i]) != '-1' and float(data['conf'][i])/100.0 >= conf_threshold:
            text += (" " + word) if text and re.match(r'[a-zA-Z0-9]', word[:1]) else word
    return text
''',

    "ocr_engine_a/tesseract_engine.py": r'''# ocr_engine_a/tesseract_engine.py
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
        for x, y, w, h in boxes:
            if w > 10 and h > 10:
                txt = dynamic_psm_recognition(cv2.copyMakeBorder(binary_n[y:y+h, x:x+w], 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255]), conf_threshold)
                if txt: txts.append(txt); cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return "\n".join(txts), img, time.time() - start
''',

    "ocr_engine_b/onnx_accelerator.py": r'''# ocr_engine_b/onnx_accelerator.py
import os, onnxruntime as ort
from paddleocr import PaddleOCR
class ONNXOCRAccelerator:
    def __init__(self, precision="INT8"):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        def get_p(m): return os.path.join(base, f"{m}_onnx_{precision.lower()}", "model.onnx")
        self.engine = PaddleOCR(det_model_dir=get_p("det"), rec_model_dir=get_p("rec"), cls_model_dir=get_p("cls"), use_angle_cls=True, use_gpu=False, use_onnx=True, show_log=False)
''',

    "post_process/box_merging.py": r'''# post_process/box_merging.py
import re
def merge_boxes(boxes, horizontal_thresh=15, height_thresh=0.1):
    if not boxes: return []
    boxes.sort(key=lambda x: x["box"][0])
    merged = []
    for item in boxes:
        if not merged: merged.append(item); continue
        last = merged[-1]
        if (item["box"][0]-last["box"][2]) <= horizontal_thresh and abs((item["box"][3]-item["box"][1])-(last["box"][3]-last["box"][1]))/max(1,last["box"][3]-last["box"][1]) <= height_thresh:
            last["box"] = [min(last["box"][0], item["box"][0]), min(last["box"][1], item["box"][1]), max(last["box"][2], item["box"][2]), max(last["box"][3], item["box"][3])]
            last["text"] += (" " if re.search(r'[a-zA-Z0-9]$', last["text"]) and re.match(r'^[a-zA-Z0-9]', item["text"]) else "") + item["text"]
        else: merged.append(item)
    return merged
''',

    "ocr_engine_b/paddle_ocr_engine.py": r'''# ocr_engine_b/paddle_ocr_engine.py
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
'''
}


def update_project():
    print(">>> 开始修正布尔值语法并同步代码...")
    os.makedirs("config", exist_ok=True)
    with open("config/params.json", 'w', encoding='utf-8') as f:
        json.dump(PARAMS_CONFIG, f, indent=4, ensure_ascii=False)

    for filepath, content in PROJECT_FILES.items():
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
        print(f"  -> [已修正] {filepath}")
    print("\n>>> 同步完成！现在可以运行 test_pipeline.py 进行测试了。")


if __name__ == "__main__":
    update_project()