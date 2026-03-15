# post_process/box_merging.py
import re
def merge_boxes(boxes, horizontal_thresh=15, height_thresh=0.1):
    if not boxes: return []
    boxes.sort(key=lambda x: (x["box"][1], x["box"][0]))
    merged = []
    for item in boxes:
        if not merged: merged.append(item); continue
        last = merged[-1]
        if (item["box"][0]-last["box"][2]) <= horizontal_thresh and abs((item["box"][3]-item["box"][1])-(last["box"][3]-last["box"][1]))/max(1,last["box"][3]-last["box"][1]) <= height_thresh:
            last["box"] = [min(last["box"][0], item["box"][0]), min(last["box"][1], item["box"][1]), max(last["box"][2], item["box"][2]), max(last["box"][3], item["box"][3])]

            last["text"] += (" " if re.search(r'[a-zA-Z0-9]$', last["text"]) and re.match(r'^[a-zA-Z0-9]', item["text"]) else "") + item["text"]
            #last["text"] += item["text"]

        else: merged.append(item)
    return merged
