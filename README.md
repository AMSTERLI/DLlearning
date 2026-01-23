Deep-Pneumonia-Detection/
├── data/                  # 数据存放
├── notebooks/             # EDA 和 实验性代码
│   └── exploration.ipynb
├── src/                   # 源代码
│   ├── dataset.py         # 数据加载与增强类
│   ├── model.py           # 定义 ResNet/DenseNet 架构
│   ├── train.py           # 训练循环
│   ├── evaluate.py        # 计算 Confusion Matrix, ROC 等
│   └── utils.py           # 辅助函数
├── visualization/         # 存放 Grad-CAM 结果图
├── grad_cam.py            # 可解释性脚本
├── requirements.txt
└── README.md              # 项目说明文档