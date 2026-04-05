# preprocess/orientation_fix.py
# 优化[优化-1]：展开压缩的链式三元表达式，符合PEP 8规范
import cv2
import numpy as np
from typing import Tuple


def fix_orientation(
    image: np.ndarray,
    max_angle_threshold: float = 15.0,
) -> Tuple[np.ndarray, float]:
    """
    基于最小外接矩形（minAreaRect）检测文本倾斜角度并自动校正。
    仅对倾斜角度在 (0.5°, max_angle_threshold°) 范围内的图像执行旋转。
    :param image: 输入图像（BGR或灰度）
    :param max_angle_threshold: 超过此角度时认为非倾斜问题，跳过校正
    :return: (校正后图像, 检测到的倾斜角度)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))

    if len(coords) == 0:
        return image, 0.0

    # 优化[优化-1]：将三元链式判断展开为清晰的 if/elif/else
    angle = cv2.minAreaRect(coords)[-1]
    if angle >= 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0

    if abs(angle) < 0.5:
        return image, 0.0
    if abs(angle) > max_angle_threshold:
        return image, angle

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    corrected = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return corrected, angle
