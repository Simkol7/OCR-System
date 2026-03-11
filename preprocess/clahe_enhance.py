# preprocess/clahe_enhance.py
import cv2
def apply_clahe(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if clip_limit <= 0: clip_limit = 2.0
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(gray_img)
