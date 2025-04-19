from torch.utils.data import Dataset
from PIL import Image
import os


class FruitVegDataset(Dataset):
    """
    自定义数据集类，用于加载水果和蔬菜图像数据集。
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        初始化数据集。

        参数:
            root_dir (str): 数据集的根目录路径。
            split (str): 数据集的划分部分（如 'train'、'validation' 或 'test'）。
            transform (callable, optional): 应用于图像的转换操作。
        """
        self.root_dir = os.path.join(root_dir, split)  # 构造数据集的根路径
        self.classes = sorted(os.listdir(self.root_dir))  # 将所有类别文件夹路径存入列表并排序
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}  # 为所有类别分配索引，存入字典

        self.transform = transform  # 图像转换操作

        self.images = []  # 存储图像路径的列表
        self.labels = []  # 存储图像标签的列表

        # 遍历每个类别文件夹，加载图像路径和标签
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)  # 指定当前类别的为工作路径
            for img_name in os.listdir(class_path):
                # 确保只加载图像文件
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_path, img_name))  # 添加图像路径
                    self.labels.append(self.class_to_idx[class_name])  # 添加对应的标签索引

    def __len__(self):
        """
        返回数据集中图像的数量。
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的图像和标签。

        参数:
            idx (int): 图像的索引。

        返回:
            image (PIL.Image): 图像对象。
            label (int): 图像的标签。
        """
        img_path = self.images[idx]  # 获取图像路径
        label = self.labels[idx]  # 获取图像标签

        image = Image.open(img_path).convert('RGB')  # 打开图像并转换为RGB格式
        if self.transform:
            image = self.transform(image)  # 应用图像转换操作

        return image, label


import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# 可视化数据增强效果
def show_augmentations(dataset, num_augments=5):
    """
    显示原始图像及其增强后的版本。

    参数:
        dataset (FruitVegDataset): 数据集对象。
        num_augments (int): 显示增强后的图像数量。
    """
    idx = random.randint(0, len(dataset)-1)  # 随机选择一个图像
    img_path = dataset.images[idx]  # 获取图像路径
    original_img = Image.open(img_path).convert('RGB')  # 打开原始图像

    plt.figure(figsize=(15, 5))  # 创建一个15x5英寸的图像窗口

    # 显示原始图像
    plt.subplot(1, num_augments+1, 1)
    plt.imshow(original_img)
    plt.title('原始图像')
    plt.axis('off')

    from dataset import train_transform
    # 显示增强后的图像
    for i in range(num_augments):
        augmented = train_transform(original_img)  # 应用增强操作
        augmented = augmented.permute(1, 2, 0).numpy()  # 将Tensor转换为NumPy数组
        augmented = (augmented * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])  # 反标准化
        augmented = np.clip(augmented, 0, 1)  # 将像素值限制在[0, 1]范围内

        plt.subplot(1, num_augments+1, i+2)
        plt.imshow(augmented)
        plt.title(f'增强图像 {i+1}')
        plt.axis('off')

    plt.tight_layout()  # 调整布局
    plt.savefig('results/augmentations.png')  # 保存图像
    plt.show()
