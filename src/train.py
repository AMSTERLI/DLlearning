import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm  # 进度条工具
import numpy as np

# 导入我们可以自己写的模块
from dataset import get_data_loaders
from model import PneumoniaNet

# ==========================================
# 配置参数 (Hyperparameters)
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # 迁移学习通常使用较小的学习率
NUM_EPOCHS = 7      # 训练轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints" # 模型保存路径

def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 Epoch"""
    model.train() # 切换到训练模式 (启用 Dropout 和 BatchNorm 更新)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用 tqdm 包装 loader，显示进度条
    loop = tqdm(loader, leave=True)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        # 1. 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 2. 反向传播与优化
        optimizer.zero_grad() # 清空梯度
        loss.backward()       # 计算梯度
        optimizer.step()      # 更新参数
        
        # 3. 统计指标
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条显示的当前 loss
        loop.set_description(f"Train")
        loop.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """验证模型性能"""
    model.eval() # 切换到评估模式 (关闭 Dropout，锁定 BatchNorm)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # 验证阶段不需要计算梯度，节省显存
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def main():
    # 1. 设置环境
    print(f"使用设备: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 2. 加载数据
    train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=BATCH_SIZE)
    print(f"类别: {classes}")
    
    # 3. 初始化模型
    # freeze_backbone=False 表示进行全网络微调，通常效果更好
    model = PneumoniaNet(num_classes=2, freeze_backbone=False).to(DEVICE)
    
    # 4. 定义损失函数和优化器
    # 针对二分类任务，CrossEntropyLoss 是标准选择
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. 开始训练循环
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training (填入完整的参数: criterion, optimizer, DEVICE)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validation (填入完整的参数: criterion, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> 模型性能提升，已保存到 {save_path}")

    print("\n训练完成！最佳验证集准确率: {:.2f}%".format(best_acc))

if __name__ == "__main__":
    main()