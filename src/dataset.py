import os
import torch
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(batch_size=32):
    # 1. 获取路径
    print("正在初始化数据路径...")
    root_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    
    base_dir = os.path.join(root_path, "chest_xray")
    if not os.path.exists(os.path.join(base_dir, "train")):
        base_dir = os.path.join(base_dir, "chest_xray")

    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test') 
    # 注意：原有的 'val' 文件夹只有16张图，太少了，我们直接忽略它，或者不使用它作为主要验证手段

    # 2. 定义增强 (Transforms)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # 训练集增强
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # 增加旋转角度到 15-20 度
        transforms.RandomRotation(20), 
        # 增加随机缩放剪裁，这对识别局部病灶非常有效
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
        # 稍微加大对比度变化，因为 X 光片明暗差异很大
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 验证/测试集只做标准化
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 3. 加载完整训练数据 (暂不放入 DataLoader)
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # 4. 关键步骤：从训练集中切分出验证集
    # 比如：90% 用于训练，10% 用于验证
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # random_split 会随机切分
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # 5. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"数据集重组完成:")
    print(f"-> 训练集: {len(train_dataset)} 张 (从原训练集切分)")
    print(f"-> 验证集: {len(val_dataset)} 张 (从原训练集切分)")
    print(f"-> 测试集: {len(test_dataset)} 张 (保持原样)")
    
    return train_loader, val_loader, test_loader, full_train_dataset.class_to_idx