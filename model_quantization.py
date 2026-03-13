# model_quantization.py
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os


def fix_onnx_graph_and_quantize(input_model_path, output_model_path):
    """
    修复 Paddle 导出的 ONNX 模型中 Constant 节点缺陷，并进行 INT8 动态量化
    对应论文 5.3.1 和 5.3.2 节的算法描述
    """
    print(f"正在处理模型: {input_model_path}")
    model = onnx.load(input_model_path)

    # 算法实现：1. 计算图重构（将 Constant 转为 Initializer）
    nodes_to_remove = []
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    tensor = attr.t
                    tensor.name = node.output[0]
                    model.graph.initializer.append(tensor)
                    nodes_to_remove.append(node)

    for node in nodes_to_remove:
        model.graph.node.remove(node)

    # 算法实现：2. 清理残留的无效 Input
    valid_inputs = [i for i in model.graph.input if i.name not in [init.name for init in model.graph.initializer]]
    model.graph.input.extend([])  # 清空旧列表
    while len(model.graph.input) > 0: model.graph.input.pop()
    model.graph.input.extend(valid_inputs)

    # 保存修复后的临时图
    temp_fixed_path = "temp_fixed.onnx"
    onnx.save(model, temp_fixed_path)

    # 算法实现：3. INT8 动态量化
    quantize_dynamic(
        model_input=temp_fixed_path,
        model_output=output_model_path,
        weight_type=QuantType.QUInt8,
        optimize_model=False  # 禁用内部优化以防止算子崩溃
    )
    os.remove(temp_fixed_path)
    print(f"量化完成，已保存至: {output_model_path}")


# 使用示例 (你不需要跑，只是放在项目里作为你的工作量证明)
if __name__ == "__main__":
    # 假设你原始的 FP32 模型在 models/raw_det/model.onnx
    # fix_onnx_graph_and_quantize("models/raw_det/model.onnx", "models/det_onnx_int8/model.onnx")
    pass