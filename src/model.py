import torch
import torch.nn as nn
from torchvision import models

class PneumoniaNet(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        """
        Args:
            num_classes: 输出分类数 (正常 vs 肺炎 = 2)
            freeze_backbone: 是否冻结预训练层的参数 (True=只训练最后分类层, False=微调整个网络)
        """
        super(PneumoniaNet, self).__init__()
        
        # 1. 加载预训练的 DenseNet-121
        # weights='DEFAULT' 会自动加载 ImageNet 上训练最好的权重
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # 2. (可选) 冻结特征提取层
        # 迁移学习初期，通常先冻结骨干网络，只训练分类头，防止破坏预训练特征
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # 3. 替换分类头 (Classifier)
        # DenseNet 的分类器层名叫 'classifier'，输入特征数通常是 1024
        num_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合，丢弃 30% 的神经元
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 简单的测试代码
# ==========================================
if __name__ == "__main__":
    # 模拟一张 224x224 的 RGB 图片 (Batch Size = 4)
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # 初始化模型
    model = PneumoniaNet(num_classes=2, freeze_backbone=True)
    
    # 前向传播
    output = model(dummy_input)
    
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}") # 应该是 [4, 2]
    print("模型构建成功！")