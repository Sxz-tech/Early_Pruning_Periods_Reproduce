import torch
import torch.nn as nn
import torch.optim as optim
from resnet import ResNet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据加载函数
def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def prune_and_rebuild_model_global(model, pruning_rate=0.9):
    """
    全局剪枝函数，并移除被剪枝的权重以确保稀疏性。
    :param model: 原始模型
    :param pruning_rate: 剪枝比例（例如 0.9 表示剪掉 90% 的参数）
    :return: 剪枝后重建的稀疏模型
    """
    # 获取全局权重的绝对值
    all_weights = torch.cat([param.data.abs().flatten() for param in model.parameters() if param.requires_grad])

    # 计算剪枝阈值
    threshold = torch.quantile(all_weights, pruning_rate)
    print(f"Global Pruning Threshold: {threshold:.6f}")

    # 初始化新的模型
    new_model = ResNet18(num_classes=10)
    pruned_shapes = {}  # 用于存储剪枝后参数的形状
    pruned_params = {}  # 用于存储剪枝后的参数

    # 剪枝权重
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 创建剪枝掩码
                mask = param.data.abs() > threshold
                pruned_params[name] = param.data[mask].clone()  # 存储未剪枝权重
                pruned_shapes[name] = param.shape  # 存储原始参数形状

    # 重新构建模型
    with torch.no_grad():
        for name, param in new_model.named_parameters():
            if name in pruned_params:
                original_shape = pruned_shapes[name]
                mask_size = pruned_params[name].numel()
                param.data.zero_()  # 初始化为零
                # 重新填充剪枝后的权重到对应位置
                param.data.view(-1)[:mask_size] = pruned_params[name].view(-1)
                param.data = param.data.view(original_shape)  # 恢复形状

    # 确保剪枝后的模型未被更新的部分保持为零
    for name, param in new_model.named_parameters():
        if name in pruned_params:
            param.requires_grad = True  # 未剪枝权重可以更新
        else:
            param.requires_grad = False  # 被剪枝部分冻结

    return new_model


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失和准确率
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 主函数
if __name__ == "__main__":
    # 超参数定义
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    NUM_EPOCHS = 200
    PRUNING_EPOCH = 175
    PRUNING_RATE = 0.9  # 剪枝比例
    GAMMA = 0.97  # 学习率指数衰减系数
    LEARNING_RATE_LOWER_BOUND = 0.001  # 学习率下限值

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # 模型、损失函数和优化器
    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

    # 训练配置
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        # 在指定 epoch 进行剪枝
        if epoch == PRUNING_EPOCH:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(LEARNING_RATE_LOWER_BOUND, current_lr)
            print(f"Pruning at epoch {epoch + 1}, setting learning rate: {new_lr:.6f}")
            model = prune_and_rebuild_model_global(model, pruning_rate=PRUNING_RATE).to(device)

            # 更新学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            optimizer = optim.SGD(model.parameters(), lr=new_lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


        scheduler.step()  # 更新学习率

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), "resnet18_pruned_cifar10.pth")
    print("训练完成，模型已保存！")


