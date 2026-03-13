text_ocr_system-master/
│
├── common/                  # [通用基础模块]
│   ├── exception_handle.py  # 定义系统全局的自定义异常类（如模型装载失败等）
│   └── utils.py             # 通用工具函数（如读取 params.json 配置项、图像合法性校验）
│
├── config/                  # [配置中心]
│   └── params.json          # 全局核心配置文件（存放阈值、引擎路径、量化精度等参数，已移除冗余的 LLM 节点）
│
├── input_module/            # [输入流管理]
│   ├── batch_loader.py      # 预留的批量图像加载器（未来扩展用）
│   └── image_loader.py      # 标准化图像读取封装
│
├── models/                  # [深度学习模型库] (注意：这些文件不会传到 GitHub)
│   ├── raw_det/ / raw_rec/  # 原始的 Paddle 导出模型（FP32精度）
│   ├── det_onnx_int8/       # INT8 量化后的检测模型（支撑论文 5.3 节性能优化）
│   └── rec_onnx_int8/       # INT8 量化后的识别模型（支撑论文 1.21s 耗时数据）
│   # 💡 之前的 frozen_east...pb 已被彻底移除，确保技术栈纯粹性
│
├── ocr_engine_a/            # [方案 A：传统机器视觉识别引擎] (对应论文第四章)
│   ├── detection/           # 形态学文本框检测子模块（膨胀、腐蚀、轮廓提取）
│   ├── recognition/         # 文本识别子模块
│   │   └── tesseract_call.py# 封装 Tesseract 调用，包含动态 PSM 宽高比匹配算法
│   └── tesseract_engine.py  # 方案 A 核心调度类（已修复 UI 实时预览所需的 get_morphology_preview 方法）
│
├── ocr_engine_b/            # [方案 B：深度学习推理引擎] (对应论文第五章核心)
│   ├── onnx_accelerator.py  # ONNX Runtime 引擎封装，负责加载 INT8 量化模型进行推理加速
│   └── paddle_ocr_engine.py # 方案 B 核心调度类，集成 DBNet 与 CRNN 的调用流程
│
├── post_process/            # [后处理与排版优化] (对应论文 5.4 节)
│   ├── box_merging.py       # 【核心】基于空间几何（水平间距与高度差）与正则语义的动态碎框合并算法
│   └── semantic_correction.py# 预留的语义纠错接口（作为论文第七章的“未来展望”）
│
├── preprocess/              # [预处理管线] (对应论文 3.2 节架构解耦)
│   ├── clahe_enhance.py     # CLAHE 自适应对比度增强算法实现
│   ├── orientation_fix.py   # 基于最小外接矩形的文本倾斜校正算法实现
│   └── preprocess_switch.py # 【核心】预处理路由分发器，解耦方案 A 和方案 B 的图像输入特征
│
├── output/                  # [测试输出目录]
│   ├── result_A.jpg         # 方案 A 识别输出的渲染图（用于对比）
│   └── result_B.jpg         # 方案 B 识别输出的渲染图（用于对比）
│
├── test/                    # [单元测试模块]
│   ├── compare_analysis.py  # 评估指标脚本（计算耗时差距、精准度）
│   └── test_case.py         # 针对不同复杂度图像的批量测试集脚本
│
├── ui/                      # [界面组件库]
│   └── ...                  # 存放分离的窗口和组件（目前集成在 main_ui.py 中协同运作）
│
├── .gitignore               # Git 忽略配置（严格屏蔽了 __pycache__ 和 models 等大文件）
├── test_image.jpg           # 系统自带的标准测试样图
├── yijian.py                # 你的开发辅助脚本（用于快速同步代码）
│
├── model_quantization.py    # 【新增 / 极度重要】计算图修复与 INT8 动态量化脚本（论文 5.3 节的直接代码证明）
├── test_pipeline.py         # 命令行全链路测试入口（一键跑出 A 和 B 的耗时对比数据，供第六章使用）
└── main_ui.py               # 【系统绝对主入口】基于 PyQt5 的高并发可视化交互界面启动文件