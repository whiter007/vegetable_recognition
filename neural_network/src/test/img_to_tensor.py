from torchvision import transforms
from PIL import Image
import numpy as np

# 创建一个简单的 480x480 图像
image = Image.new('RGB', (480, 480), (125, 94, 63))  # 创建一个单色的图像

# 定义变换
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),         # 转换为 Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

# 应用变换
transformed_image = val_transform(image)

# 打印输出张量
print(transformed_image)
print(transformed_image.shape)
transformed_image = transformed_image.unsqueeze(0)  # 添加批次维度
print(transformed_image)
print(transformed_image.shape)