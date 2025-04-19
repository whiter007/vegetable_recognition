import torch
import torch.nn as nn
import torch.optim as optim
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
        self.batch_size = 32  # 批量大小
        self.epochs = 30  # 训练周期数
        self.dropout = 0.3  # Dropout比率
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检测是否有可用的GPU，如果有则使用GPU，否则使用CPU
        # 定义数据集增强策略
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像大小调整为224x224

            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(15),  # 随机旋转15度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机调整亮度、对比度和饱和度

            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet数据集的归一化参数：前者是红、绿、蓝三个通道的均值，后者是标准差
            ])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像大小调整为224x224
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet数据集的归一化参数：前者是红、绿、蓝三个通道的均值，后者是标准差
            ])

if __name__ == '__main__':

    """ ----- 【预处理】----- """
    os.makedirs('results', exist_ok=True)  # 创建用于保存输出结果的文件夹，且当文件夹已存在时不会报错
    set_seed()  # 调用set_seed函数设置随机种子
    data_path = "../"  # 数据集路径
    config = Config()  # 初始化配置

    """ ----- 【数据处理、数据集准备、数据加载器】----- """
    # explore_data(data_path)
    train_dataset = FruitVegDataset(data_path,  # 数据集根目录
                                    'train',  # 训练集目录名
                                    config.train_transform  # 数据增强策略
                                    )  # 训练数据集
    val_dataset = FruitVegDataset(data_path,  # 数据集根目录
                                  'validation',  # 验证集目录名
                                  config.val_transform  # 数据增强策略
                                  )  # 验证数据集
    # show_augmentations(train_dataset)
    train_loader = DataLoader(train_dataset,  # 继承了Dataset的数据集类
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=2
                              )  # 训练数据加载器
    val_loader = DataLoader(val_dataset,  # 继承了Dataset的数据集类
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=2
                            )  # 验证数据加载器


    """ -----【模型的设备选择与加载】----- """
    # 初始化模型
    model = FruitVegCNN(num_classes=len(train_dataset.classes)).to(config.device)  # 初始化模型并移动到设备上

    """ ----- 【模型训练】 ----- """
    # sample_image, _ = train_dataset[0]  # 获取一个样本图像
    # visualize_feature_maps(model, sample_image)  # 可视化特征图
    # 训练设置
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # 定义优化器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)  # 学习率调度器
    # 训练循环
    best_val_acc = 0  # 最佳验证准确率
    history = {
        'train_loss': [], 'train_acc': [],  # 训练损失和准确率
        'val_loss': [], 'val_acc': []  # 验证损失和准确率
    }

    print("\n开始训练...")
    for epoch in range(config.epochs):
        print(f'\n周期 {epoch+1}/{config.epochs}')

        # 训练一个周期，并返回训练损失和准确率
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device)

        # 验证模型，并返回训练损失和准确率
        val_loss, val_acc = validate(
            model, val_loader, criterion, config.device)

        # 更新学习率
        scheduler.step(val_loss)

        # 保存训练记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%')

        # 每5个周期绘制一次训练过程
        # if (epoch + 1) % 5 == 0:
        #     plot_training_progress(history)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'新的最佳验证准确率: {best_val_acc:.2f}%')
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'best_acc': best_val_acc,
            # }, 'results/best_model.pth')  # 保存模型状态
            # 保存为 .safetensors 文件
            model_state_dict = model.state_dict()  # 获取模型的状态字典
            save_file(model_state_dict, "first_model.safetensors")
            print("模型已保存为 first_model.safetensors")

    # 最终训练可视化
    plot_training_progress(history)