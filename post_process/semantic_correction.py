# post_process/semantic_correction.py
import re

class SemanticCorrector:
    """语义纠错类（基础版：正则+简单替换，进阶可接LLM）"""
    def __init__(self, mode="dict"):
        self.mode = mode
        self.common_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千万年月日"

    def correct(self, text):
        if self.mode == "dict":
            return self._dict_correction(text)
        return text

    def _dict_correction(self, text):
        # 简单过滤演示，实际可接入 Levenshtein 距离
        filtered = re.sub(rf"[^{self.common_chars}\u4e00-\u9fa5]", "", text)
        return filtered if filtered else text
