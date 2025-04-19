from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

import torch
import numpy as np
import random
# 设置随机种子以确保结果可复现
def set_seed(seed=32):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    # 设置PyTorch CUDA的随机种子
    torch.cuda.manual_seed_all(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置Python内置random模块的随机种子
    random.seed(seed)
    # 设置PyTorch的确定性模式
    torch.backends.cudnn.deterministic = True
    # 关闭PyTorch的性能优化模式
    torch.backends.cudnn.benchmark = False


import os
from PIL import Image
import matplotlib.pyplot as plt
def explore_data(data_path):
    """探索并可视化数据集"""
    print("\n探索数据集结构：")
    print("-" * 50)

    # 定义数据集的划分部分（训练集、验证集、测试集）
    splits = ['train', 'validation', 'test']
    for split in splits:
        # 构造当前划分部分的路径
        split_path = os.path.join(data_path, split)
        if os.path.exists(split_path):
            # 获取当前划分部分的所有类别（文件夹名）
            classes = sorted(os.listdir(split_path))
            # 统计当前划分部分的总图像数量
            total_images = sum(len(os.listdir(os.path.join(split_path, cls)))
                               for cls in classes)

            print(f"\n{split.capitalize()} Set:")
            print(f"类别数量: {len(classes)}")
            print(f"总图像数量: {total_images}")
            print(f"示例类别: {', '.join(classes[:5])}...")

    # 可视化样本图像
    print("\n正在可视化样本图像...")
    train_path = os.path.join(data_path, 'train')
    classes = sorted(os.listdir(train_path))

    # 创建一个15x10英寸的图像窗口
    plt.figure(figsize=(15, 10))
    for i in range(9):
        # 随机选择一个类别
        class_name = random.choice(classes)
        class_path = os.path.join(train_path, class_name)
        # 随机选择该类别下的一个图像
        img_name = random.choice(os.listdir(class_path))
        img_path = os.path.join(class_path, img_name)

        # 打开图像
        img = Image.open(img_path)
        # 将图像绘制到子图中
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        plt.title(f'类别: {class_name}')
        plt.axis('off')  # 关闭坐标轴

    # 调整布局
    plt.tight_layout()
    # 保存可视化结果
    plt.savefig('results/sample_images.png')
    plt.show()




# import matplotlib.pyplot as plt
# 可视化特征图的函数
def visualize_feature_maps(model, sample_image, device):
    """
    可视化每个卷积模块后的特征图。

    参数:
        model (FruitVegCNN): 模型对象。
        sample_image (Tensor): 输入图像张量。
    """
    model.eval()  # 将模型设置为评估模式

    # 获取每个卷积模块后的特征图
    feature_maps = []
    x = sample_image.unsqueeze(0).to(device)  # 添加批次维度并移动到设备上

    for block in model.features:
        x = block(x)  # 通过每个卷积模块
        feature_maps.append(x.detach().cpu())  # 保存特征图并移回CPU

    # 绘制特征图
    plt.figure(figsize=(15, 10))  # 创建一个15x10英寸的图像窗口
    for i, fmap in enumerate(feature_maps):
        # 只绘制每个模块的前6个通道
        fmap = fmap[0][:6].permute(1, 2, 0)  # 提取前6个通道并调整维度
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())  # 归一化特征图

        for j in range(min(6, fmap.shape[-1])):  # 遍历通道
            plt.subplot(5, 6, i*6 + j + 1)  # 创建子图
            plt.imshow(fmap[:, :, j], cmap='viridis')  # 绘制特征图
            plt.title(f'模块 {i+1}, 通道 {j+1}')
            plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()  # 调整布局
    plt.savefig('results/feature_maps.png')  # 保存图像
    plt.show()



# import matplotlib.pyplot as plt
# 绘制训练过程的函数
def plot_training_progress(history):
    """
    绘制并保存训练过程的损失和准确率曲线。

    参数:
        history (dict): 训练过程的记录，包含训练和验证的损失及准确率。
    """
    plt.figure(figsize=(12, 4))  # 创建一个12x4英寸的图像窗口

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失历史')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率历史')
    plt.xlabel('周期')
    plt.ylabel('准确率 (%)')
    plt.legend()

    plt.tight_layout()  # 调整布局
    plt.savefig('results/training_progress.png')  # 保存图像
    plt.show()