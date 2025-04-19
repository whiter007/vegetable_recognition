import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# 导入本地模块
from utils import *
from dataset import *
from model import *
from train import *


# 定义优化后的训练配置
class Config:
    def __init__(self):
        self.image_size = 256  # 图像尺寸增加到256
        self.batch_size = 16   # 使用较小的批量大小以提高泛化能力
        self.learning_rate = 3e-4  # 学习率
        self.weight_decay = 0.01  # 权重衰减
        self.epochs = 50  # 训练周期数
        self.dropout = 0.3  # Dropout比率
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检测是否有可用的GPU，如果有则使用GPU，否则使用CPU
        # 定义数据集增强策略
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 将图像大小调整为256x256

            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪到224x224
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomVerticalFlip(),  # 随机垂直翻转
            transforms.RandomRotation(20),  # 随机旋转20度
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 随机仿射变换
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整颜色
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),  # 30%概率应用高斯模糊

            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet数据集的归一化参数：前者是红、绿、蓝三个通道的均值，后者是标准差
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像大小调整为224x224
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet数据集的归一化参数：前者是红、绿、蓝三个通道的均值，后者是标准差
        ])

from safetensors.torch import save_file

if __name__ == '__main__':

    """ 预处理 """
    os.makedirs('results', exist_ok=True)  # 创建用于保存输出结果的文件夹，并且文件夹已存在时不会报错
    set_seed()  # 调用set_seed函数设置随机种子
    data_path = "../"  # 数据集路径
    config = Config()  # 初始化配置

    """ 数据处理、数据集准备、数据加载器 """
    # explore_data(data_path)
    train_dataset = FruitVegDataset(data_path,  # 数据集根目录
                                    'train',  # 训练集目录名
                                    config.train_transform  # 数据增强策略
                                    )  # 训练数据集
    val_dataset = FruitVegDataset(data_path,  # 数据集根目录
                                  'validation',  # 验证集目录名
                                  config.val_transform  # 数据增强策略
                                  )  # 验证数据集
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True
                              )  # 训练数据加载器
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True
                            )  # 验证数据加载器
    print(train_dataset.classes)  # 打印训练数据集的类别

    """ 模型加载 """
    # 初始化模型
    model = OptimizedCNN(num_classes=len(train_dataset.classes)).to(config.device)  # 初始化模型并移动到设备
    # model.modules()  # 打印模型结构
    # print(model)  # 打印模型结构

    """ 模型训练 """
    history = train_with_optimization(model, train_loader, val_loader, config, config.device)  # 开始训练

    # 最终训练可视化
    plot_training_progress(history)

    # print(train_dataset.root_dir)
    # print(train_dataset.classes)
