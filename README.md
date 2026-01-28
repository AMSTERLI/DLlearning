# 🫁 Deep Pneumonia Detection (DenseNet121)

基于 PyTorch 的深度学习项目，旨在利用迁移学习（**DenseNet121**）从胸部 X 光片（Chest X-Ray）中自动检测肺炎。

本项目针对 Kaggle 原始数据集的验证集过小问题进行了优化，实现了自动化的数据重组、增强与训练监控。

## ✨ 核心特性

* **⚡️ 自动化数据流**: 集成 `kagglehub`，首次运行时自动下载数据集，无需手动配置路径。
* **🔄 智能数据集重组**: 代码自动将原始训练集按 **90:10** 比例重新划分为训练集与验证集，解决原版验证集只有 16 张图的问题。
* **🛡️ 鲁棒的数据增强**: 训练时应用随机旋转 (+/- 20°)、随机裁剪缩放 (RandomResizedCrop) 和色彩抖动，提升模型泛化能力。
* **🧠 迁移学习**: 基于 ImageNet 预训练的 **DenseNet121** 进行全网络微调。DenseNet121 在医学影像分析中表现优异（类似 CheXNet 架构）。

## 📂 项目结构

```text
Deep-Pneumonia-Detection/
├── data/                  # 数据存放 (kagglehub 自动下载或手动存放)
├── checkpoints/           # 训练过程中保存的最佳模型权重 (.pth)
├── notebooks/             # Jupyter Notebooks (EDA 和 实验性代码)
├── src/                   # 核心源代码
│   ├── dataset.py         # 数据集加载、自动切分逻辑与 Transforms 定义
│   ├── model.py           # 模型定义 (DenseNet121 修改版)
│   ├── train.py           # 训练主循环、验证与早停逻辑
│   ├── evaluate.py        # 模型评估指标计算
│   └── utils.py           # 辅助工具函数
├── visualization/         # 存放 Grad-CAM 热力图结果
├── grad_cam.py            # Grad-CAM 可视化脚本
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明文档
```

## 🛠️ 环境安装

建议使用 Python 3.8+ 和 CUDA 环境。

克隆仓库

Bash

git clone [https://github.com/your-username/Deep-Pneumonia-Detection.git](https://github.com/your-username/Deep-Pneumonia-Detection.git)
cd Deep-Pneumonia-Detection
安装依赖

Bash

pip install -r requirements.txt
(核心依赖: torch, torchvision, kagglehub, tqdm, matplotlib, livelossplot)

## 🚀 快速开始

1. 训练模型
直接运行训练脚本。脚本会检测数据是否存在，不存在则自动下载。

Bash

python src/train.py
默认配置:

Batch Size: 32

Epochs: 10 

Learning Rate: 1e-4

输出: 最佳模型权重将保存在 checkpoints/best_model.pth。

2. 评估模型 (可选)
加载最佳权重并在测试集（624张图片）上计算准确率和混淆矩阵。

Bash

python src/evaluate.py


## 📊 数据集详情

使用 Chest X-Ray Images (Pneumonia) 数据集。

代码运行时会自动执行以下划分策略：

训练集 (Train): ~4,694 张 (应用强增强)

验证集 (Val): ~522 张 (仅标准化，用于监控模型性能)

测试集 (Test): 624 张 (保持官方原样，用于最终测试)

## 🧠 模型架构
Backbone: DenseNet121 (Pretrained on ImageNet)

特点: 密集连接网络，能有效缓解梯度消失，特征复用率高，非常适合纹理细节丰富的医学图像。

修改: 替换最后的分类层 (classifier) 以输出 2 个类别 (Normal vs Pneumonia)。

Loss Function: CrossEntropyLoss

Optimizer: Adam