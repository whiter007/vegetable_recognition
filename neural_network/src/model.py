import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    卷积模块，包含两层卷积层、批量归一化、ReLU激活函数和最大池化。
    """
    def __init__(self, in_channels, out_channels):
        """
        初始化卷积模块。

        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3卷积，填充为1保持尺寸
            nn.BatchNorm2d(out_channels),  # 批量归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 第二层3x3卷积
            nn.BatchNorm2d(out_channels),  # 批量归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=2)  # 2x2最大池化
        )

    def forward(self, x):
        return self.conv(x)


class FruitVegCNN(nn.Module):
    """
    水果和蔬菜分类的卷积神经网络模型。
    """
    def __init__(self, num_classes):
        """
        初始化模型。

        参数:
            num_classes (int): 分类的类别数。
        """
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 64),  # 输入通道为3（RGB图像），输出通道为64
            ConvBlock(64, 128),  # 输入通道为64，输出通道为128
            ConvBlock(128, 256),  # 输入通道为128，输出通道为256
            ConvBlock(256, 512),  # 输入通道为256，输出通道为512
            ConvBlock(512, 512)  # 输入通道为512，输出通道为512
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化，将特征图大小压缩为1x1
            nn.Flatten(),  # 展平特征
            nn.Dropout(0.5),  # Dropout，防止过拟合
            nn.Linear(512, 256),  # 全连接层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Dropout(0.5),  # Dropout
            nn.Linear(256, num_classes)  # 输出层
        )

    def forward(self, x):
        return self.classifier(self.features(x)) #先提取特征再分类

# 定义优化后的模型架构
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 使用预训练的ResNet50作为骨干网络
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0',
                                     'resnet50', pretrained=True)

        # 冻结早期层
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False

        # 修改分类器部分
        num_features = self.backbone.fc.in_features  # 获取ResNet50的特征维度
        self.backbone.fc = nn.Sequential(  #替换预训练模型的分类器
            nn.Linear(num_features, 1024),  # 第一层全连接层，输入特征维度为num_features，隐藏层一特征维度为1024
            nn.BatchNorm1d(1024),  # 批量归一化
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Dropout(0.3),  # Dropout

            nn.Linear(1024, 512),  # 第二层全连接层，隐藏层二特征维度为512
            nn.BatchNorm1d(512),  # 批量归一化
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Dropout(0.3),  # Dropout

            nn.Linear(512, num_classes)  # 输出层
        )

    def forward(self, x):
        return self.backbone(x)