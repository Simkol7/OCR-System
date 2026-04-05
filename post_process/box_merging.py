# post_process/box_merging.py
import re
from typing import List, Dict, Any, Optional
from common.utils import load_config


def merge_boxes(
    boxes: List[Dict[str, Any]],
    horizontal_thresh: Optional[float] = None,
    height_thresh: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    合并水平相邻、高度相近的文本框，解决DBNet碎框检测问题。
    阈值优先使用传入参数；未传入时从 config/params.json 的 box_merging 节读取。
    修复[普通-1]：合并阈值现在实际从配置文件中加载，而非硬编码。
    """
    # 从配置文件读取合并阈值（修复[普通-1]）
    if horizontal_thresh is None or height_thresh is None:
        config = load_config().get("box_merging", {})
        if horizontal_thresh is None:
            horizontal_thresh = config.get("horizontal_threshold", 15)
        if height_thresh is None:
            height_thresh = config.get("height_threshold", 0.1)

    if not boxes:
        return []

    boxes.sort(key=lambda x: (x["box"][1], x["box"][0]))
    merged = []

    for item in boxes:
        if not merged:
            merged.append(item)
            continue

        last = merged[-1]
        gap = item["box"][0] - last["box"][2]
        last_h = max(1, last["box"][3] - last["box"][1])
        item_h = item["box"][3] - item["box"][1]
        height_diff_ratio = abs(item_h - last_h) / last_h

        if gap <= horizontal_thresh and height_diff_ratio <= height_thresh:
            # 扩展合并框的边界至两框的并集
            last["box"] = [
                min(last["box"][0], item["box"][0]),
                min(last["box"][1], item["box"][1]),
                max(last["box"][2], item["box"][2]),
                max(last["box"][3], item["box"][3]),
            ]
            # 中英文混排时在单词间补空格，修复[普通-6]：删除遗留的注释废代码
            sep = " " if (
                re.search(r'[a-zA-Z0-9]$', last["text"])
                and re.match(r'^[a-zA-Z0-9]', item["text"])
            ) else ""
            last["text"] += sep + item["text"]
        else:
            merged.append(item)

    return merged
