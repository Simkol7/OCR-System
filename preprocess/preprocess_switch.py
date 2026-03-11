# preprocess/preprocess_switch.py
import cv2
from .clahe_enhance import apply_clahe
from .orientation_fix import fix_orientation
from common.utils import load_config
def run_preprocess(image, scheme="B", **kwargs):
    config = load_config().get("preprocess", {})
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    enhanced = apply_clahe(gray, clip_limit=config.get("clahe_clip_limit", 2.0))
    fixed_img, _ = fix_orientation(enhanced, max_angle_threshold=config.get("angle_fix_threshold", 15))
    if scheme == "B": return fixed_img
    binary = cv2.adaptiveThreshold(cv2.medianBlur(fixed_img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    eroded = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (int(kwargs.get("erode_size", 2)), int(kwargs.get("erode_size", 2)))))
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (int(kwargs.get("dilate_x", 15)), int(kwargs.get("dilate_y", 3)))))
    return dilated, cv2.bitwise_not(binary)
