# preprocess/orientation_fix.py
import cv2, numpy as np
def fix_orientation(image, max_angle_threshold=15.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0: return image, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    angle = angle - 90.0 if angle >= 45.0 else (angle + 90.0 if angle < -45.0 else angle)
    if abs(angle) < 0.5: return image, 0.0
    if abs(angle) > max_angle_threshold: return image, angle
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)), angle
