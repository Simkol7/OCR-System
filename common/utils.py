# common/utils.py
import json, os
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "params.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}
def check_image_validity(image, path=""):
    if image is None or image.size == 0: raise Exception(f"图像无效: {path}")
