import torch
import sys

print("1. Python 路径:", sys.executable)
print("2. PyTorch 版本:", torch.__version__)
print("3. CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("4. 当前显卡设备:", torch.cuda.get_device_name(0))
    print("   恭喜！你可以使用显卡加速训练了！")
else:
    print("   警告：目前只能使用 CPU。请检查安装步骤。")