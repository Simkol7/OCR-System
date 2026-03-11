# common/exception_handle.py
class OCRBaseError(Exception): pass
class ImageReadError(OCRBaseError): pass
class ImageEmptyError(OCRBaseError): pass
class ParameterError(OCRBaseError): pass
class OCRInferenceError(OCRBaseError): pass
