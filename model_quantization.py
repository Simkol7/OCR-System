# model_quantization.py
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def fix_onnx_graph_and_quantize(input_model_path, output_model_path):
    """
    修复 Paddle 导出的 ONNX 模型中 Constant 节点缺陷，并进行 INT8 动态量化。
    对应论文 5.3.1 和 5.3.2 节的算法描述。

    算法步骤：
    1. 计算图重构：将 Constant 节点转为 Initializer，解决 ONNX Runtime 兼容性问题
    2. 清理残留无效 Input 节点
    3. INT8 动态量化，压缩权重体积并加速推理
    """
    print(f"正在处理模型: {input_model_path}")
    model = onnx.load(input_model_path)

    # 算法实现：1. 计算图重构（将 Constant 节点转为 Initializer）
    nodes_to_remove = []
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    tensor = attr.t
                    tensor.name = node.output[0]
                    model.graph.initializer.append(tensor)
                    nodes_to_remove.append(node)

    for node in nodes_to_remove:
        model.graph.node.remove(node)

    # 算法实现：2. 清理残留的无效 Input（已被提升为 Initializer 的节点）
    initializer_names = {init.name for init in model.graph.initializer}
    valid_inputs = [i for i in model.graph.input if i.name not in initializer_names]
    # 修复[优化-1]：用 del 替代冗余的 extend([]) + while pop() 操作
    del model.graph.input[:]
    model.graph.input.extend(valid_inputs)

    # 保存修复后的临时计算图，量化完成后用 try/finally 确保临时文件被清理
    temp_fixed_path = "temp_fixed.onnx"
    try:
        onnx.save(model, temp_fixed_path)

        # 算法实现：3. INT8 动态量化
        quantize_dynamic(
            model_input=temp_fixed_path,
            model_output=output_model_path,
            weight_type=QuantType.QUInt8,
            optimize_model=False,  # 禁用内部优化以防止算子崩溃
        )
    finally:
        # 无论量化是否成功，都清理中间临时文件
        if os.path.exists(temp_fixed_path):
            os.remove(temp_fixed_path)

    print(f"量化完成，已保存至: {output_model_path}")


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # 使用说明：
    # 前置条件：已将 PaddleOCR PP-OCRv4 模型导出为 ONNX FP32 格式，放置于
    #   models/det_onnx_fp32/model.onnx
    #   models/rec_onnx_fp32/model.onnx
    #   models/cls_onnx_fp32/model.onnx
    # 执行方式：取消下方三行注释后，在项目根目录运行 `python model_quantization.py`
    # 输出结果：INT8 量化模型将写入对应的 *_onnx_int8/ 目录
    # -----------------------------------------------------------------------
    fix_onnx_graph_and_quantize(
        "models/det_onnx_fp32/model.onnx",
        "models/det_onnx_int8/model.onnx",
    )
    fix_onnx_graph_and_quantize(
        "models/rec_onnx_fp32/model.onnx",
        "models/rec_onnx_int8/model.onnx",
    )
    fix_onnx_graph_and_quantize(
        "models/cls_onnx_fp32/model.onnx",
        "models/cls_onnx_int8/model.onnx",
    )
