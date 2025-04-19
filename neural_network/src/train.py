import torch
from tqdm import tqdm


# 训练一个周期的函数
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练模型一个周期。

    参数:
        model (nn.Module): 模型对象。
        train_loader (DataLoader): 训练数据加载器。
        criterion (nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        device (torch.device): 训练设备（CPU或GPU）。
    """
    model.train()  # 调用继承的方法，将模型设置为训练模式
    running_loss = 0.0  # 累计损失
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数量

    pbar = tqdm(train_loader, desc='Training')  # 创建进度条
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到指定设备

        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失

        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        running_loss += loss.item()  # 累计损失
        _, predicted = outputs.max(1)  # 获取预测结果
        total += labels.size(0)  # 累计总样本数
        correct += predicted.eq(labels).sum().item()  # 累计正确预测数

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',  # 显示当前批次损失
            'acc': f'{100.*correct/total:.2f}%'  # 显示当前训练准确率
        })

    return running_loss / len(train_loader), 100. * correct / total  # 返回平均损失和准确率


# 验证函数
def validate(model, val_loader, criterion, device):
    """
    验证模型性能。

    参数:
        model (nn.Module): 模型对象。
        val_loader (DataLoader): 验证数据加载器。
        criterion (nn.Module): 损失函数。
        device (torch.device): 验证设备（CPU或GPU）。
    """
    model.eval()  # 将模型设置为评估模式
    running_loss = 0.0  # 累计损失
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数量

    with torch.no_grad():  # 关闭梯度计算
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到指定设备

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            running_loss += loss.item()  # 累计损失
            _, predicted = outputs.max(1)  # 获取预测结果
            total += labels.size(0)  # 累计总样本数
            correct += predicted.eq(labels).sum().item()  # 累计正确预测数

    return running_loss / len(val_loader), 100. * correct / total  # 返回平均损失和准确率



import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from safetensors.torch import save_file, load_file
# 定义优化后的训练函数
def train_with_optimization(model, train_loader, val_loader, config, device):
    """
    使用优化配置训练模型。

    参数:
        model (nn.Module): 模型对象。
        train_loader (DataLoader): 训练数据加载器。
        val_loader (DataLoader): 验证数据加载器。
        config (OptimizedConfig): 训练配置。
    """
    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑的交叉熵损失
    # 优化器
    optimizer = optim.AdamW(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)  # 使用AdamW优化器
    # 学习率调度器
    # 使用One Cycle Learning Rate Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # 使用GradScaler进行混合精度训练
    scaler = GradScaler()

    history = {
        'train_loss': [], 'train_acc': [],  # 训练损失和准确率
        'val_loss': [], 'val_acc': []  # 验证损失和准确率
    }

    best_val_acc = 0  # 记录最佳验证准确率

    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')  # 创建进度条
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到设备

            optimizer.zero_grad()  # 清空梯度

            # 使用混合精度训练
            with autocast():
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失

            scaler.scale(loss).backward()  # 反向传播
            scaler.step(optimizer)  # 更新优化器
            scaler.update()  # 更新GradScaler
            scheduler.step()  # 更新学习率

            train_loss += loss.item()  # 累计损失
            _, predicted = outputs.max(1)  # 获取预测结果
            total += labels.size(0)  # 累计总样本数
            correct += predicted.eq(labels).sum().item()  # 累计正确预测数

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',  # 显示当前批次损失
                'acc': f'{100.*correct/total:.2f}%',  # 显示当前训练准确率
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'  # 显示当前学习率
            })

        train_acc = 100. * correct / total  # 计算训练准确率
        train_loss = train_loss / len(train_loader)  # 计算平均训练损失

        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():  # 关闭梯度计算
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到设备
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失

                val_loss += loss.item()  # 累计损失
                _, predicted = outputs.max(1)  # 获取预测结果
                total += labels.size(0)  # 累计总样本数
                correct += predicted.eq(labels).sum().item()  # 累计正确预测数

        val_acc = 100. * correct / total  # 计算验证准确率
        val_loss = val_loss / len(val_loader)  # 计算平均验证损失

        # 保存训练记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'\n周期 {epoch+1}/{config.epochs}:')
        print(f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%')

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
            # 保存完整模型
            # torch.save(model, 'model.pt')  # 保存模型状态
            # 保存模型为 .safetensors 格式
            model_state_dict = model.state_dict()  # 获取模型的状态字典
            save_file(model_state_dict, "model.safetensors")
            print("模型已保存为 model.safetensors")

    return history