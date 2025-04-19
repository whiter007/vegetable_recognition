import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os

#导入本地模块
from utils import *
from model import *

# 加载保存的模型
def load_model():
    # 尝试加载模型文件
    try:
        checkpoint = torch.load('optimized_model.pth')  # 加载模型检查点
        model = OptimizedCNN(num_classes=36)  # 创建模型实例，类别数与训练时一致
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型状态字典
        model.eval()  # 设置模型为评估模式
        print("模型加载成功！")
        return model
    except FileNotFoundError:
        print("未找到模型文件 'optimized_model.pth'！")
        return None

# 预测函数
def predict_image(url, model):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])

    # 从URL加载图像
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')  # 打开并转换为RGB图像

    # 应用图像变换
    input_tensor = transform(image).unsqueeze(0)  # 增加批量维度

    # 进行预测
    with torch.no_grad():  # 禁用梯度计算
        output = model(input_tensor)  # 模型推理
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # 计算softmax概率

        # 获取前5个预测结果
        top_probs, top_indices = torch.topk(probabilities, 5)

    # 显示结果
    plt.figure(figsize=(12, 4))

    # 显示输入图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('输入图像')
    plt.axis('off')

    # 显示预测结果
    plt.subplot(1, 2, 2)
    classes = sorted(os.listdir("../train"))  # 获取类别列表
    y_pos = range(5)
    plt.barh(y_pos, [prob.item() * 100 for prob in top_probs])  # 水平柱状图
    plt.yticks(y_pos, [classes[idx] for idx in top_indices])  # 设置类别标签
    plt.xlabel('概率 (%)')
    plt.title('前5个预测结果')

    plt.tight_layout()
    plt.show()

    # 打印预测结果
    print("\n预测结果：")
    print("-" * 30)
    for i in range(5):
        print(f"{classes[top_indices[i]]:20s}: {top_probs[i]*100:.2f}%")

# 加载模型
model = load_model()

# 使用示例
predict_image('https://pngimg.com/uploads/watermelon/watermelon_PNG2639.png', model)