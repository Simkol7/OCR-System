# 双引擎文本识别系统

基于 PyQt5 的双引擎 OCR 对比系统，集成传统机器视觉（OpenCV + Tesseract）与深度学习（PaddleOCR + ONNX Runtime 加载离线 INT8 量化模型）两种识别方案，支持可视化参数调节与结果实时对比。本项目为本科毕业设计（计算机方向）。

---

## 系统架构

```
text_ocr_system-master/
│
├── common/                        # 通用基础模块
│   ├── exception_handle.py        # 自定义异常层次（OCRBaseError → ImageReadError 等）
│   └── utils.py                   # load_config() / check_image_validity()
│
├── config/
│   └── params.json                # 全局配置（CLAHE阈值、形态学参数、合并阈值等）
│
├── preprocess/                    # 预处理管线（对应论文 3.2 节）
│   ├── preprocess_switch.py       # 路由分发：方案A返回灰度图，方案B返回BGR图
│   ├── clahe_enhance.py           # CLAHE 自适应对比度增强
│   └── orientation_fix.py         # 基于 minAreaRect 的文本倾斜校正
│
├── ocr_engine_a/                  # 方案 A：传统机器视觉引擎（对应论文第四章）
│   ├── tesseract_engine.py        # AlgorithmA 调度类，含 get_morphology_preview()
│   └── recognition/
│       └── tesseract_call.py      # Tesseract 封装，动态 PSM 宽高比匹配
│
├── ocr_engine_b/                  # 方案 B：深度学习引擎（对应论文第五章）
│   ├── paddle_ocr_engine.py       # AlgorithmB 调度类，集成 DBNet + CRNN
│   └── onnx_accelerator.py        # ONNX Runtime 加载离线量化模型
│
├── post_process/                  # 后处理模块（对应论文 5.4 节）
│   ├── box_merging.py             # 基于空间几何的碎框合并算法
│   └── semantic_correction.py     # 字符白名单过滤后处理
│
├── models/                        # 深度学习模型（不上传 GitHub）
│   ├── raw_det/                   # PP-OCRv4 检测模型（FP32）
│   ├── raw_rec/                   # PP-OCRv4 识别模型（FP32）
│   ├── raw_cls/                   # 方向分类模型（FP32）
│   ├── det_onnx_fp32/             # 检测模型 ONNX 导出版
│   ├── det_onnx_int8/             # 检测模型 INT8 量化版（论文 5.3 节）
│   ├── rec_onnx_fp32/             # 识别模型 ONNX 导出版
│   ├── rec_onnx_int8/             # 识别模型 INT8 量化版
│   ├── cls_onnx_fp32/             # 分类模型 ONNX 导出版
│   └── cls_onnx_int8/             # 分类模型 INT8 量化版
│
├── test/                          # 测试模块
│   ├── test_case.py               # 批量测试集脚本，输出 output/test_results.csv
│   ├── compare_analysis.py        # 精度与耗时对比分析脚本
│   ├── test_pipeline.py           # 命令行全链路测试入口（A/B 双方案耗时对比）
│   └── test_image.jpg             # 系统自带标准测试样图
│
├── test_dataset/                  # 测试图像集
│   └── basic_2.png                # 示例测试图
│
├── output/                        # 测试输出目录（自动创建）
│   ├── result_A.jpg               # 方案 A 识别可视化结果
│   └── result_B.jpg               # 方案 B 识别可视化结果
│
├── main_ui.py                     # 【主入口】PyQt5 可视化界面
├── model_quantization.py          # 计算图修复 + INT8 动态量化脚本（论文 5.3 节）
└── requirements.txt               # Python 依赖清单
```

---

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.9+ | 推荐 3.10 |
| PyQt5 | 5.15.0 | GUI 框架 |
| paddlepaddle | 2.5.0 | PaddleOCR 运行时 |
| paddleocr | 2.7.0 | OCR 推理框架 |
| onnxruntime | 1.16.0 | ONNX 推理加速 |
| onnx | 1.14.0 | 模型导出与量化 |
| opencv-python | 4.8.0 | 图像预处理 |
| numpy | 1.24.0 | 数值计算 |
| pytesseract | 0.3.10 | Tesseract Python 封装 |

---

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/<your-repo>/text_ocr_system-master.git
cd text_ocr_system-master
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 Tesseract OCR 可执行程序（方案 A 必须）

- **Windows**：从 [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) 下载安装包，安装时勾选中文语言包（`chi_sim`）
- 安装完成后，在 `config/params.json` 中配置路径：

```json
{
    "ocr_a": {
        "tesseract_cmd_path": "D:\\Tesseract-OCR\\tesseract.exe",
        "lang": "chi_sim+eng"
    }
}
```

### 4. 准备模型文件（方案 B 必须）

将 PaddleOCR PP-OCRv4 模型放置到 `models/` 目录下（参考上方目录结构）。
如需生成 INT8 量化模型，运行：

```bash
python model_quantization.py
```

> 注意：`models/` 目录已加入 `.gitignore`，不会上传至 GitHub，需自行准备。

---

## 使用方式

### 启动图形界面（推荐）

```bash
python main_ui.py
```

界面功能说明：
- **加载图像**：支持 JPG、PNG、BMP 等常见格式
- **选择引擎**：左侧单选框切换方案 A（Tesseract）或方案 B（PaddleOCR）
- **参数调节**：
  - 方案 A：可实时调整形态学膨胀/腐蚀参数，预览中间结果
  - 方案 B：可调节 `unclip_ratio` 控制文本框扩张幅度
- **开始识别**：后台线程异步执行，界面不阻塞
- **导出结果**：将识别文本保存为 `.txt` 文件

### 命令行全链路测试

```bash
python test/test_pipeline.py
```

对 `test/test_image.jpg` 同时运行方案 A 和方案 B，输出各自耗时与识别文本，并将可视化结果保存至 `output/` 目录。

### 批量测试与精度分析

```bash
# 批量测试（生成 output/test_results.csv）
python test/test_case.py

# 对比分析（生成 output/accuracy_results.csv）
python test/compare_analysis.py
```

---

## 配置说明

所有可调参数集中在 `config/params.json`，修改后重启程序生效：

```json
{
    "preprocess": {
        "clahe_clip_limit": 2.0,
        "angle_fix_threshold": 15,
        "morph_erode_size": 2,
        "morph_dilate_x": 15,
        "morph_dilate_y": 3
    },
    "ocr_a": {
        "tesseract_cmd_path": "D:\\Tesseract-OCR\\tesseract.exe",
        "lang": "chi_sim+eng"
    },
    "ocr_b": {
        "use_angle_cls": true,
        "lang": "ch",
        "model_precision": "INT8",
        "inference_engine": "onnx"
    },
    "box_merging": {
        "horizontal_threshold": 15,
        "height_threshold": 0.1
    }
}
```

---

## 技术方案对比

| 维度 | 方案 A（传统视觉） | 方案 B（深度学习） |
|------|------------------|------------------|
| 检测方式 | OpenCV 形态学 + 轮廓提取 | DBNet 可微分二值化 |
| 识别引擎 | Tesseract LSTM | PaddleOCR CRNN |
| 推理加速 | 无 | ONNX Runtime 加载离线 INT8 量化模型 |
| 中文支持 | chi_sim 语言包 | PP-OCRv4 中文模型 |
| 适用场景 | 简单印刷体、英文 | 中英混排、复杂版面 |
| 平均耗时 | 较快 | 更准确，INT8 加速后耗时可控 |

---

## 项目模块依赖关系

```
main_ui.py
    ├── ocr_engine_a/tesseract_engine.py (AlgorithmA)
    │       ├── preprocess/preprocess_switch.py
    │       │       ├── preprocess/clahe_enhance.py
    │       │       └── preprocess/orientation_fix.py
    │       ├── recognition/tesseract_call.py
    │       └── post_process/semantic_correction.py
    └── ocr_engine_b/paddle_ocr_engine.py (AlgorithmB)
            ├── preprocess/preprocess_switch.py
            ├── ocr_engine_b/onnx_accelerator.py
            ├── post_process/box_merging.py
            └── post_process/semantic_correction.py
```

---

## 未来展望

- **文本后处理升级**：当前 `semantic_correction.py` 为字符白名单过滤基础版，可替换为基于词典或语言模型的纠错方案
- **批量导出**：扩展 `test/test_case.py` 支持多图批量处理与汇总报告输出
