import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from safetensors.torch import load_file
from model import OptimizedCNN

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSES = [
    "apple苹果",
    "banana香蕉",
    "beetroot甜菜根",
    "bell pepper柿子椒",
    "cabbage卷心菜",
    "capsicum灯笼椒",
    "carrot胡萝卜",
    "cauliflower花椰菜",
    "chilli pepper辣椒",
    "corn玉米",
    "cucumber黄瓜",
    "eggplant茄子",
    "garlic大蒜",
    "ginger姜",
    "grapes葡萄",
    "jalepeno墨西哥辣椒",
    "kiwi猕猴桃",
    "lemon柠檬",
    "lettuce生菜",
    "mango芒果",
    "onion洋葱",
    "orange橙子",
    "paprika红椒",
    "pear梨",
    "peas豌豆",
    "pineapple菠萝",
    "pomegranate石榴",
    "potato土豆",
    "raddish萝卜",
    "soy beans大豆",
    "spinach菠菜",
    "sweetcorn甜玉米",
    "sweetpotato红薯",
    "tomato西红柿",
    "turnip芜菁",
    "watermelon西瓜",
];

def predict(image_path: str) -> str:
    """
    图片预测函数
    :param image_path: 图片路径
    :return: 预测结果字符串
    """
    # 加载图片并进行预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 添加批次维度

    # 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA is available: {cuda_available}")

    # 加载模型
    device = torch.device('cuda' if cuda_available else 'cpu')
    print(f"Using device: {device}")

    model = OptimizedCNN(num_classes=len(CLASSES)).to(device)
    # 加载权重
    weight_path = "model.safetensors"  # 替换为你的权重文件路径
    try:
        state_dict = load_file(weight_path)
        model.load_state_dict(state_dict)
        print("模型权重加载成功！")
    except FileNotFoundError:
        print(f"错误：未找到 {weight_path} 文件。")
    except Exception as e:
        print(f"加载权重时出现错误：{e}")

    model.eval()

    # 模型推理
    with torch.no_grad():
        logits = model(image.to(device))
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        top5_prob, top5_indices = probabilities.topk(5)

    # 处理预测结果
    predict_result = "\n".join(
        [f"{CLASSES[idx.item()]}: {prob.item() * 100:.2f}%" for idx, prob in zip(top5_indices[0], top5_prob[0])]
    )
    return predict_result