# ocr_engine_a/recognition/tesseract_call.py
import re
import pytesseract
from common.utils import load_config
from common.exception_handle import OCRInferenceError


def dynamic_psm_recognition(roi, conf_threshold, config=None):
    """
    根据ROI区域的宽高比动态选择Tesseract PSM模式，提升单行/单字符/多行场景的识别精度。
    - 接近正方形 (0.8 < w/h < 1.2)：PSM 10（单字符模式）
    - 较高区域 (h > 50 且 w/h < 2.5)：PSM 3（全自动分页模式）
    - 其他宽条形区域：PSM 7（单行文本模式）
    :param roi: 预处理后的文本区域图像
    :param conf_threshold: 置信度过滤阈值（0~1）
    :param config: ocr_a 配置字典；由调用方传入以避免每个 ROI 重复读磁盘，
                   为 None 时自动从 params.json 加载（兜底）
    :return: 识别到的文本字符串
    :raises OCRInferenceError: Tesseract调用失败时抛出
    """
    # 调用方传入 config 则直接使用，否则兜底加载（避免每 ROI 重复 I/O）
    if config is None:
        config = load_config().get("ocr_a", {})

    # 仅在配置了路径时覆盖，避免 None 赋值导致 pytesseract 找不到可执行文件
    tesseract_path = config.get("tesseract_cmd_path")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    h, w = roi.shape[:2]
    aspect_ratio = w / h if h > 0 else 0

    # 动态PSM模式选择
    if h > 0 and 0.8 < aspect_ratio < 1.2:
        psm = 10   # 单字符
    elif h > 50 and aspect_ratio < 2.5:
        psm = 3    # 全自动分页
    else:
        psm = 7    # 单行文本

    # lang 配置缺失时使用中英文混合语言包兜底，防止静默退化为纯英文识别
    lang = config.get("lang") or "chi_sim+eng"

    try:
        data = pytesseract.image_to_data(
            roi,
            lang=lang,
            config=f"--oem 3 --psm {psm}",
            output_type=pytesseract.Output.DICT,
        )
    except Exception as e:
        raise OCRInferenceError(f"Tesseract调用失败（psm={psm}）: {e}") from e

    text = ""
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = data["conf"][i]
        if not word or str(conf) == "-1":
            continue
        if float(conf) / 100.0 < conf_threshold:
            continue
        # 英文/数字单词之间补空格，中文字符直接拼接
        if text and re.match(r"[a-zA-Z0-9]", word[:1]):
            text += " " + word
        else:
            text += word

    return text
