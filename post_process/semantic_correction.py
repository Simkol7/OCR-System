# post_process/semantic_correction.py
import re


class SemanticCorrector:
    """
    文本后处理类（基础版：正则字符白名单过滤，预留LLM接入接口）。
    通过过滤非法字符去除OCR引擎常见的噪声输出，保留中英文、数字及空格。
    """

    def __init__(self, mode: str = "dict"):
        """
        :param mode: 纠错模式，当前支持 "dict"（正则过滤），预留扩展
        """
        self.mode = mode
        # 保留：英文字母、数字、空格（保持词间间距，避免英文词连读）
        # 中文字符通过正则 \u4e00-\u9fa5 统一覆盖，无需在此枚举
        self.valid_chars = r"0-9A-Za-z \u4e00-\u9fa5"

    def correct(self, text: str) -> str:
        """对输入文本应用纠错策略，返回纠错后的字符串。"""
        if not isinstance(text, str):
            return ""
        if self.mode == "dict":
            return self._dict_correction(text)
        return text

    def _dict_correction(self, text: str) -> str:
        """
        基于字符白名单的正则过滤。
        保留中文、英文字母、数字和空格，过滤其余噪声字符。
        若过滤后结果为空，返回原文本作为安全兜底。
        """
        filtered = re.sub(rf"[^{self.valid_chars}]", "", text)
        return filtered if filtered.strip() else text
