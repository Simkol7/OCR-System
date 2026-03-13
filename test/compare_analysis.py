# test/compare_analysis.py
import os
import re
import sys
import difflib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr_engine_a.tesseract_engine import AlgorithmA
from ocr_engine_b.paddle_ocr_engine import AlgorithmB


def calculate_accuracy(pred_text, true_text):
    """
    基于字符覆盖召回率（Character Recall）的学术级评估
    剔除排版乱序与全半角标点符号的干扰，纯粹评估 OCR 引擎对字符的识别精度
    """
    if not true_text:
        return 0.0

    # 1. 极致净化：使用正则剔除所有标点、空格、换行，只保留汉字、英文字母和数字
    # \w 匹配字母数字下划线，\u4e00-\u9fa5 匹配所有汉字
    pred_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', pred_text).lower()
    true_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', true_text).lower()

    if not true_clean:
        return 0.0

    # 2. 计算字符覆盖率：容忍多栏排版导致的乱序问题
    matched_count = 0
    pred_list = list(pred_clean)

    for char in true_clean:
        if char in pred_list:
            matched_count += 1
            pred_list.remove(char)  # 匹配过就移除，防止重复计算

    # 计算召回率：正确识别出的有效字符数 / 真实存在的有效字符数
    accuracy = (matched_count / len(true_clean)) * 100.0
    return min(accuracy, 100.0)  # 封顶 100%


def run_accuracy_evaluation(gt_file, img_dir):
    if not os.path.exists(gt_file):
        print(f"[错误] 找不到真实标签文件: {gt_file}")
        return

    print("========== 启动 OCR 字符级准确率(Accuracy)客观评估 ==========")
    algo_a = AlgorithmA()
    algo_b = AlgorithmB()
    algo_b.load_model()

    total_acc_a = 0.0
    total_acc_b = 0.0
    valid_count = 0

    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue

            img_name, true_text = line.split('|', 1)
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                print(f"[警告] 图片不存在: {img_path}，跳过该项。")
                continue

            print(f"\n正在评估: {img_name}")

            # --- 方案 A 推理 ---
            try:
                text_a, _, _ = algo_a.detect_and_recognize(img_path, conf_threshold=0.5, erode_size=2, dilate_x=15,
                                                           dilate_y=3)
                acc_a = calculate_accuracy(text_a, true_text)
            except Exception:
                acc_a = 0.0

            # --- 方案 B 推理 ---
            try:
                text_b, _, _ = algo_b.detect_and_recognize(img_path, conf_threshold=0.5, unclip_ratio=1.5)
                acc_b = calculate_accuracy(text_b, true_text)
            except Exception:
                acc_b = 0.0

            print(f"  -> 真实标签长度: {len(true_text)} 字符")
            print(f"  -> [方案 A] 准确率: {acc_a:.2f}%")
            print(f"  -> [方案 B] 准确率: {acc_b:.2f}%")

            total_acc_a += acc_a
            total_acc_b += acc_b
            valid_count += 1

    if valid_count > 0:
        avg_a = total_acc_a / valid_count
        avg_b = total_acc_b / valid_count
        print("\n========== 客观评估结论 ==========")
        print(f"评估样本数: {valid_count} 张图像")
        print(f"【传统方案 A】平均字符准确率: {avg_a:.2f}%")
        print(f"【深度学习方案 B】平均字符准确率: {avg_b:.2f}%")
    else:
        print("\n未找到有效的评估样本。")


if __name__ == "__main__":
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.txt")
    image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_dataset")
    run_accuracy_evaluation(gt_path, image_dir)