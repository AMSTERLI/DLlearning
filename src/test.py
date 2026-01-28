import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# 导入我们自己的模块
from model import PneumoniaNet
from dataset import get_data_loaders

# ==========================================
# 配置参数
# ==========================================
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model.pth"
VIS_DIR = "visualization" # 结果保存目录

def test_process(model, loader, device):
    """
    专门用于测试的函数
    返回: 真实标签列表, 预测标签列表
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

def save_confusion_matrix(cm, classes, output_path):
    """绘制并保存混淆矩阵热力图"""
    plt.figure(figsize=(8, 6))
    # 使用 seaborn 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # 保存图片而不是显示
    print(f"正在保存混淆矩阵图到: {output_path}")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close() # 关闭图表以释放内存

def main():
    print(f"使用设备: {DEVICE}")
    
    # 0. 创建结果保存目录
    os.makedirs(VIS_DIR, exist_ok=True)
    # 生成一个带时间戳的文件名前缀，防止覆盖之前的测试结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 准备数据
    print("正在加载测试数据...")
    # 我们只需要 test_loader
    _, _, test_loader, classes_idx = get_data_loaders(batch_size=BATCH_SIZE)
    # 反转字典获取类别名称列表: ['NORMAL', 'PNEUMONIA']
    idx_to_class = {v: k for k, v in classes_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # 2. 初始化模型并加载权重
    model = PneumoniaNet(num_classes=len(class_names)).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"正在加载模型权重: {CHECKPOINT_PATH}")
        # map_location 确保跨设备兼容性
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print(f"错误: 找不到权重文件 {CHECKPOINT_PATH}，请先运行 train.py")
        return

    # 3. 开始测试
    print("开始在测试集上评估...")
    y_true, y_pred = test_process(model, test_loader, DEVICE)
    
    # 4. 计算指标
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    acc = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)

    # 5. 保存文本报告 (保存到 visualization 文件夹)
    report_path = os.path.join(VIS_DIR, f"report_{timestamp}.txt")
    
    report_content = f"测试时间: {timestamp}\n"
    report_content += f"模型权重: {CHECKPOINT_PATH}\n"
    report_content += "="*50 + "\n"
    report_content += f"最终测试集准确率 (Accuracy): {acc*100:.2f}%\n"
    report_content += "="*50 + "\n\n"
    report_content += "=== 混淆矩阵 (Confusion Matrix) ===\n"
    report_content += str(cm) + "\n\n"
    report_content += "=== 详细分类报告 (Classification Report) ===\n"
    report_content += cr + "\n"

    # 将内容写入txt文件
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n测试报告已保存到: {report_path}")
    # 同时也打印到控制台方便查看
    print(report_content)

    # 6. 保存混淆矩阵图片 (保存到 visualization 文件夹)
    cm_img_path = os.path.join(VIS_DIR, f"confusion_matrix_{timestamp}.png")
    try:
        save_confusion_matrix(cm, class_names, cm_img_path)
        print("测试完成！")
    except Exception as e:
        print(f"绘图失败 (可能是缺少 seaborn 库): {e}")

if __name__ == "__main__":
    main()