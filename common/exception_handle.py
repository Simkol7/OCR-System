# common/exception_handle.py
# 优化[优化-1]：展开压缩的类定义，符合PEP 8规范


class OCRBaseError(Exception):
    """OCR系统异常基类，所有业务异常均继承自此类。"""
    pass


class ImageReadError(OCRBaseError):
    """图像文件读取失败异常（路径不存在或格式不支持）。"""
    pass


class ImageEmptyError(OCRBaseError):
    """图像内容为空或尺寸无效异常。"""
    pass


class ParameterError(OCRBaseError):
    """调用参数非法或超出有效范围异常。"""
    pass


class OCRInferenceError(OCRBaseError):
    """OCR模型推理过程中发生的运行时异常。"""
    pass
