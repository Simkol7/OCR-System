# preprocess/clahe_enhance.py
# 优化[优化-1]：展开压缩的函数体，符合PEP 8规范
import cv2
import numpy as np
from typing import Tuple


def apply_clahe(
    gray_img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    对灰度图像应用CLAHE（限制对比度自适应直方图均衡化），增强局部对比度。
    :param gray_img: 输入灰度图
    :param clip_limit: 对比度限制阈值，防止过度放大噪声
    :param tile_grid_size: 分块网格大小
    :return: 增强后的灰度图
    """
    if clip_limit <= 0:
        clip_limit = 2.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)
