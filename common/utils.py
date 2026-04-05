# common/utils.py
import json
import os
from typing import Any, Dict

from common.exception_handle import ImageEmptyError, ImageReadError


def load_config() -> Dict[str, Any]:
    """加载 config/params.json 配置文件，失败时返回空字典并打印警告。"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "params.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[警告] 配置文件加载失败: {e}，使用默认参数")
        return {}


def check_image_validity(image: Any, path: str = "") -> None:
    """
    检查图像是否有效，无效时抛出对应异常。
    :param image: numpy ndarray 图像对象
    :param path: 图像来源路径（用于错误信息）
    :raises ImageReadError: image 为 None，即文件路径不存在或格式不支持时抛出
    :raises ImageEmptyError: image 尺寸为零，即文件损坏或内容为空时抛出
    """
    if image is None:
        raise ImageReadError(f"图像读取失败，请检查文件路径或格式: {path}")
    if image.size == 0:
        raise ImageEmptyError(f"图像无效或内容为空: {path}")
