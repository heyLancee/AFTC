import torch
import torch.onnx
import argparse
from td3 import Actor

def convert_pth_to_onnx(pth_path, onnx_path, input_shape, device='cpu'):
    """
    将PyTorch模型转换为ONNX格式
    
    Args:
        pth_path: PyTorch模型路径(.pth文件)
        onnx_path: 输出的ONNX模型路径
        input_shape: 输入张量的形状
        device: 运行设备 ('cpu' 或 'cuda')
    """
    # 创建模型实例
    model = Actor(state_dim=input_shape[1], action_dim=4, max_action=1.0, hidden_size=[256, 256])
    
    # 加载模型权重
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 创建随机输入张量
    dummy_input = torch.randn(input_shape, device=device)
    
    # 导出ONNX模型
    torch.onnx.export(
        model,                  # 要转换的模型
        dummy_input,           # 模型输入
        onnx_path,            # 保存路径
        export_params=True,    # 存储训练好的参数权重
        opset_version=11,      # ONNX算子集版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],     # 输入节点的名称
        output_names=['output'],   # 输出节点的名称
        dynamic_axes={             # 动态尺寸的设置
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"模型已成功转换为ONNX格式并保存到: {onnx_path}")

    # 验证导出的模型
    verify_onnx(model, dummy_input, onnx_path)

def verify_onnx(pytorch_model, dummy_input, onnx_path):
    """
    验证转换后的ONNX模型输出是否与PyTorch模型一致
    """
    import onnxruntime
    import numpy as np
    
    # PyTorch模型推理
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input)
    
    # ONNX模型推理
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # 比较输出结果
    np.testing.assert_allclose(
        pytorch_output.cpu().numpy(), 
        ort_output, 
        rtol=1e-03, 
        atol=1e-05
    )
    print("PyTorch和ONNX模型输出一致，验证通过！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将PyTorch模型转换为ONNX格式")
    parser.add_argument('--pth_path', type=str, required=True, help='PyTorch模型路径')
    parser.add_argument('--onnx_path', type=str, required=True, help='ONNX模型保存路径')
    parser.add_argument('--state_dim', type=int, required=True, help='状态维度')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备 (cpu 或 cuda)')
    
    args = parser.parse_args()
    
    # 设置输入形状
    input_shape = (args.batch_size, args.state_dim)
    
    # 执行转换
    convert_pth_to_onnx(args.pth_path, args.onnx_path, input_shape, args.device)
