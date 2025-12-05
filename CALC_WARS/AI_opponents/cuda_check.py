# 新建文件 check_cuda.py
import torch
import sys

print("="*50)
print("🔥 CUDA 环境诊断")
print("="*50)

# 1. PyTorch版本和CUDA编译版本
print(f"\n1. PyTorch版本: {torch.__version__}")
print(f"   CUDA编译版本: {torch.version.cuda}")

# 2. CUDA是否可用
print(f"\n2. torch.cuda.is_available(): {torch.cuda.is_available()}")

# 3. 设备数量
if torch.cuda.is_available():
    print(f"   GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("   ❌ 警告: CUDA不可用！")

# 4. 当前设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n3. 默认设备: {device}")

# 5. 创建张量并监控显存
print(f"\n4. 显存测试:")
print(f"   创建前显存: {torch.cuda.memory_allocated()/1e6:.1f} MB")

# 创建大张量强制占用显存
test_tensor = torch.randn(4096, 121).to(device)
print(f"   创建后显存: {torch.cuda.memory_allocated()/1e6:.1f} MB")
print(f"   张量设备: {test_tensor.device}")

# 6. 模型测试
print(f"\n5. 模型测试:")
class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(121, 29)
    
    def forward(self, x):
        return self.fc(x)

net = DummyNet().to(device)
print(f"   模型参数设备: {next(net.parameters()).device}")
print(f"   模型前向传播显存: {torch.cuda.memory_allocated()/1e6:.1f} MB")

out = net(test_tensor)
print(f"   输出设备: {out.device}")
print(f"   最终显存占用: {torch.cuda.memory_allocated()/1e6:.1f} MB")

# 7. 检查环境变量
print(f"\n6. 环境变量:")
import os
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
print(f"   PATH包含cuda: {'cuda' in os.environ.get('PATH', '').lower()}")

# 8. 如果CUDA不可用，检查conda
if not torch.cuda.is_available():
    print(f"\n❌ 检测到CUDA不可用！执行修复:")
    print(f"   你的PyTorch可能是CPU版本")
    print(f"   尝试运行: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "="*50)