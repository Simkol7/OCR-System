# ocr_engine_a/recognition/tesseract_call.py
import pytesseract, re
from common.utils import load_config
def dynamic_psm_recognition(roi, conf_threshold):
    config = load_config().get("ocr_a", {})
    pytesseract.pytesseract.tesseract_cmd = config.get("tesseract_cmd_path")
    h, w = roi.shape[:2]
    psm = 10 if h>0 and 0.8<w/h<1.2 else (3 if h>50 and w/h<2.5 else 7)
    data = pytesseract.image_to_data(roi, lang=config.get('lang'), config=f'--oem 3 --psm {psm}', output_type=pytesseract.Output.DICT)
    text = ""
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word and str(data['conf'][i]) != '-1' and float(data['conf'][i])/100.0 >= conf_threshold:
            text += (" " + word) if text and re.match(r'[a-zA-Z0-9]', word[:1]) else word
    return text
