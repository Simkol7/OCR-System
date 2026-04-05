# preprocess/preprocess_switch.py
# 优化[优化-1]：展开超长压缩行，符合PEP 8规范
import cv2
from .clahe_enhance import apply_clahe
from .orientation_fix import fix_orientation
from common.utils import load_config
from common.exception_handle import ParameterError


def run_preprocess(image, scheme: str = "B", **kwargs):
    """
    预处理路由分发器，根据方案类型执行不同的图像处理管线。
    - scheme="B"：CLAHE增强 → 倾斜校正 → 转回BGR（供PaddleOCR使用）
    - scheme="A"：CLAHE增强 → 倾斜校正 → 自适应二值化 → 腐蚀+膨胀（供轮廓检测使用）
    :return: scheme="B"时返回BGR图像；scheme="A"时返回 (膨胀图, 二值化反转图) 元组
    """
    if scheme not in ("A", "B"):
        raise ParameterError(f"不支持的预处理方案: '{scheme}'，有效值为 'A' 或 'B'")

    config = load_config().get("preprocess", {})

    # 统一转灰度后执行增强与校正
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    enhanced = apply_clahe(gray, clip_limit=config.get("clahe_clip_limit", 2.0))
    fixed_img, _ = fix_orientation(enhanced, max_angle_threshold=config.get("angle_fix_threshold", 15))

    if scheme == "B":
        # 修复[致命-1]：将灰度图转回BGR三通道，确保PaddleOCR的DBNet检测模型接收正确格式
        return cv2.cvtColor(fixed_img, cv2.COLOR_GRAY2BGR)

    # 方案A专用管线：自适应二值化 → 腐蚀去噪 → 膨胀连接
    blurred = cv2.medianBlur(fixed_img, 3)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,   # blockSize: 11×11 像素邻域计算自适应阈值，适配中等字体大小
        2,    # C: 从加权均值中减去的常数，调节二值化灵敏度
    )

    erode_ksize = int(kwargs.get("erode_size", 2))
    eroded = cv2.erode(
        binary,
        cv2.getStructuringElement(cv2.MORPH_RECT, (erode_ksize, erode_ksize)),
    )

    dilate_ksize_x = int(kwargs.get("dilate_x", 15))
    dilate_ksize_y = int(kwargs.get("dilate_y", 3))
    dilated = cv2.dilate(
        eroded,
        cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ksize_x, dilate_ksize_y)),
    )

    return dilated, cv2.bitwise_not(binary)
