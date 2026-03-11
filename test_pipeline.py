# test_pipeline.py
import os
import cv2
import numpy as np
from ocr_engine_a.tesseract_engine import AlgorithmA
from ocr_engine_b.paddle_ocr_engine import AlgorithmB


def run_full_pipeline_test(image_path):
    if not os.path.exists(image_path):
        print(f"[错误] 测试图像不存在: {image_path}")
        return

    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)

    print(f"========== 开始全链路测试 | 测试图像: {image_path} ==========")

    # ---------------------------------------------------------
    # 测试方案 A (OpenCV + Tesseract)
    # ---------------------------------------------------------
    print("\n[---> 方案 A 测试启动 <---]")
    try:
        algo_a = AlgorithmA()
        # 传入默认形态学参数
        text_a, img_a, time_a = algo_a.detect_and_recognize(
            image_path,
            conf_threshold=0.5,
            erode_size=2,
            dilate_x=15,
            dilate_y=3
        )
        print(f"【方案 A 耗时】: {time_a:.4f} 秒")
        print(f"【方案 A 识别结果】:\n{text_a if text_a.strip() else '<未识别到任何文本>'}")

        if img_a is not None:
            out_path_a = os.path.join("output", "result_A.jpg")
            cv2.imwrite(out_path_a, img_a)
            print(f"【方案 A 可视化】: 结果已保存至 {out_path_a}")

    except Exception as e:
        print(f"[异常] 方案 A 运行失败: {e}")

    # ---------------------------------------------------------
    # 测试方案 B (PaddleOCR 重构进阶版)
    # ---------------------------------------------------------
    print("\n[---> 方案 B 测试启动 <---]")
    try:
        algo_b = AlgorithmB()
        text_b, img_b, time_b = algo_b.detect_and_recognize(
            image_path,
            conf_threshold=0.5,
            unclip_ratio=1.5
        )
        print(f"【方案 B 耗时】: {time_b:.4f} 秒")
        print(f"【方案 B 识别结果】:\n{text_b if text_b.strip() else '<未识别到任何文本>'}")

        if img_b is not None:
            out_path_b = os.path.join("output", "result_B.jpg")
            cv2.imwrite(out_path_b, img_b)
            print(f"【方案 B 可视化】: 结果已保存至 {out_path_b}")

    except Exception as e:
        print(f"[异常] 方案 B 运行失败: {e}")


if __name__ == "__main__":
    # 指定测试图片路径
    test_img = "test_image.jpg"

    # 客观环境处理：如果当前目录下没有名为 test_image.jpg 的图片，自动生成一张包含基础英文文本的白底黑字图进行连通性验证
    if not os.path.exists(test_img):
        print(f"未找到测试图像 '{test_img}'，系统正在自动生成一张基础测试图...")
        dummy_img = np.ones((300, 800, 3), dtype=np.uint8) * 255
        # 写入几行存在倾斜风险或尺寸差异的文本
        cv2.putText(dummy_img, "SYSTEM TEST 2026", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(dummy_img, "Pipeline Integration Check", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(test_img, dummy_img)

    run_full_pipeline_test(test_img)