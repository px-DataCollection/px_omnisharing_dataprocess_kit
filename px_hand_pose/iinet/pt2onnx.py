import torch
import torch.onnx
import numpy as np
import argparse
import os
import sys
sys.path.append('./IINet')
import warnings

from modules.disp_model import DispModel

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning)

# 导入自定义模块
import options

import torchvision.transforms as transforms
from PIL import Image

# 定义预处理参数
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 创建模型包装器类，解决输入字典问题
class DispModelWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
    
    def forward(self, left, right):
        # 将输入打包为字典格式
        inputs = {'left': left, 'right': right}
        outputs = self.model(inputs)
        # 只返回我们需要的输出
        return outputs['disp_pred_s0']

def convert_to_onnx(opts, onnx_path, input_shape=(720, 1280), dynamic_axes=True):
    """
    将IINet模型转换为ONNX格式
    
    参数:
    opts: 配置选项对象
    onnx_path: 输出ONNX文件路径(.onnx)
    input_shape: 输入图像形状(height, width)
    dynamic_axes: 是否启用动态轴(处理可变尺寸输入)
    """
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 初始化模型
    original_model = DispModel(opts).to(device)
    
    # 3. 加载模型权重
    if not os.path.isfile(opts.load_weights_from_checkpoint):
        raise RuntimeError(f"=> no checkpoint found at '{opts.load_weights_from_checkpoint}'")
    
    # 安全加载模型权重
    checkpoint = torch.load(opts.load_weights_from_checkpoint, weights_only=True)
    pretrained_dict = checkpoint['state_dict']
    original_model.load_state_dict(pretrained_dict, strict=False)  # 使用strict=False忽略不匹配的键
    original_model.eval()  # 设置为评估模式
    
    # 4. 创建包装器模型
    model = DispModelWrapper(original_model).to(device)
    model.eval()
    
    # 5. 准备输入数据 - 创建随机图像
    height, width = input_shape
    left_pic = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    right_pic = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # 6. 应用与测试函数相同的预处理
    # 填充处理
    h, w, c = left_pic.shape
    top_pad = (32 - (h % 32)) % 32
    right_pad = (32 - (w % 32)) % 32
    
    if not (top_pad == 0 and right_pad == 0):
        left_pic = np.lib.pad(left_pic, ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant', constant_values=0)
        right_pic = np.lib.pad(right_pic, ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant', constant_values=0)
    
    # 转换和标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    left_tensor = transform(left_pic).unsqueeze(0).to(device)
    right_tensor = transform(right_pic).unsqueeze(0).to(device)
    
    print(f"Tensor shape (input): left={left_tensor.shape}, right={right_tensor.shape}")
    
    # 7. 导出ONNX模型
    input_names = ["left", "right"]  # 输入节点名称
    output_names = ["disp_pred_s0"]  # 输出节点名称
    
    # 8. 执行导出
    torch.onnx.export(
        model,                      # 要导出的模型
        (left_tensor, right_tensor),  # 模型输入（两个张量）
        onnx_path,                  # 输出ONNX文件路径
        export_params=True,         # 导出模型参数
        opset_version=16,           # ONNX算子集版本
        do_constant_folding=True,   # 优化常量折叠
        input_names=input_names,    # 输入节点名称
        output_names=output_names,  # 输出节点名称
        # dynamic_axes=dynamic_axes_params,  # 动态轴配置
        verbose=True                # 显示详细信息
    )
    
    print(f"Model has been exported as ONNX: {onnx_path}")
    
    # 9. 验证导出是否成功
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification complete")

if __name__ == "__main__":
    # 初始化选项处理器
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    opts = option_handler.options
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Export IINet to ONNX')
    parser.add_argument('--onnx_path', type=str, default='iinet.onnx', help='Output ONNX file path')
    parser.add_argument('--height', type=int, default=720, help='Input image height')
    parser.add_argument('--width', type=int, default=1280, help='Input image width')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sceneflow.tar', help='Path to model checkpoint')
    args = parser.parse_args()
    
    # 更新opts中的检查点路径
    opts.load_weights_from_checkpoint = args.checkpoint
    
    # 调用转换函数
    convert_to_onnx(
        opts=opts,
        onnx_path=args.onnx_path,
        input_shape=(args.height, args.width),
        dynamic_axes=True
    )