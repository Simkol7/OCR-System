# test/test_case.py
import os
import sys
import time
import csv
import cv2
import numpy as np

# 确保能导入外层模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_engine_a.tesseract_engine import AlgorithmA
from ocr_engine_b.paddle_ocr_engine import AlgorithmB


def run_batch_test(test_dir="test_dataset", output_csv="test_results.csv"):
    if not os.path.exists(test_dir):
        print(f"[错误] 测试集目录 '{test_dir}' 不存在，请先创建并放入测试图片。")
        return

    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print(f"[错误] 目录 '{test_dir}' 中没有找到任何图像文件。")
        return

    print(f"========== 启动批量对比测试 | 共 {len(image_files)} 张图像 ==========")

    algo_a = AlgorithmA()
    algo_b = AlgorithmB()

    # 预热模型
    print("[系统] 正在预热深度学习引擎...")
    algo_b.load_model()

    results = []
    # 创建一个临时文件用于存放压缩后的图片
    temp_img_path = os.path.join(test_dir, "_temp_resized.jpg")

    for idx, img_name in enumerate(image_files, 1):
        original_img_path = os.path.join(test_dir, img_name)
        print(f"\n[{idx}/{len(image_files)}] 正在测试: {img_name}")

        # ===================================================
        # 核心抢救逻辑：动态等比例压缩超大图像，防止 CPU 显存/内存爆炸
        # ===================================================
        img_check = cv2.imdecode(np.fromfile(original_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_check is None:
            print(f"  -> [跳过] 无法读取图像: {img_name}")
            continue

        h, w = img_check.shape[:2]
        max_len = 1280  # 限制最长边为 1280 像素 (足以看清文字，又不会卡死)

        if max(h, w) > max_len:
            scale = max_len / max(h, w)
            img_resized = cv2.resize(img_check, (int(w * scale), int(h * scale)))
            # 保存为临时文件供引擎读取
            cv2.imencode('.jpg', img_resized)[1].tofile(temp_img_path)
            target_path = temp_img_path
        else:
            target_path = original_img_path
        # ===================================================

        # --- 测试方案 A ---
        try:
            text_a, _, time_a = algo_a.detect_and_recognize(target_path, conf_threshold=0.5, erode_size=2, dilate_x=15,
                                                            dilate_y=3)
            # 净化文本输出，把换行替换为空格，截取前60个字符展示
            clean_text_a = text_a.replace('\n', ' ').strip()
            display_a = clean_text_a[:60] + "..." if len(clean_text_a) > 60 else clean_text_a
        except Exception as e:
            print(f"  -> 方案 A 失败: {e}")
            time_a, clean_text_a, display_a = 0.0, "", "识别失败"

        # --- 测试方案 B ---
        try:
            text_b, _, time_b = algo_b.detect_and_recognize(target_path, conf_threshold=0.5, unclip_ratio=1.5)
            clean_text_b = text_b.replace('\n', ' ').strip()
            display_b = clean_text_b[:60] + "..." if len(clean_text_b) > 60 else clean_text_b
        except Exception as e:
            print(f"  -> 方案 B 失败: {e}")
            time_b, clean_text_b, display_b = 0.0, "", "识别失败"

        # 打印耗时和识别出的文本片段
        print(f"  -> [方案A] 耗时: {time_a:.3f}s | 文本: {display_a if display_a else '<空>'}")
        print(f"  -> [方案B] 耗时: {time_b:.3f}s | 文本: {display_b if display_b else '<空>'}")

        results.append({
            "Image": img_name,
            "Time_A(s)": round(time_a, 3),
            "Time_B(s)": round(time_b, 3),
            "CharCount_A": len(clean_text_a),
            "CharCount_B": len(clean_text_b)
        })

    # 清理临时文件
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    # --- 写入 CSV 报告 ---
    output_csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_csv)
    with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "Time_A(s)", "Time_B(s)", "CharCount_A", "CharCount_B"])
        writer.writeheader()
        writer.writerows(results)

    # --- 计算客观平均值 ---
    avg_time_a = sum(r["Time_A(s)"] for r in results) / max(1, len(results))
    avg_time_b = sum(r["Time_B(s)"] for r in results) / max(1, len(results))
    print("\n========== 测试完成 ==========")
    print(f"【客观统计】方案 A 平均耗时: {avg_time_a:.3f} 秒")
    print(f"【客观统计】方案 B 平均耗时: {avg_time_b:.3f} 秒")
    print(f"详细数据已保存至根目录下的: {output_csv}")


if __name__ == "__main__":
    # 使用相对路径指向外层的 test_dataset
    run_batch_test(test_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_dataset"),
                   output_csv="test_results.csv")