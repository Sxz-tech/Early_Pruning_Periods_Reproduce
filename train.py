import torch
import torch.nn as nn
import torch.optim as optim
from resnet import ResNet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle

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

# 计算 KL_uniform 指标
def compute_kl_uniform(model, data_loader, device, max_batches=10):
    """
    计算 KL_uniform 指标
    :param model: 当前模型
    :param data_loader: 数据加载器
    :param device: GPU 或 CPU
    :param max_batches: 计算的最大批次数
    :return: KL_uniform 指标（标量）
    """
    kl_uniform = 0.0
    model.eval()  # 模型切换到评估模式
    total_samples = 0
    num_classes = 10  # CIFAR-10 的类别数

    for batch_idx, (inputs, _) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        inputs = inputs.to(device)

        # 前向传播，计算 softmax 概率分布
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # 计算 KL_uniform = - (1/C) * sum_y log(p(y|x))
        kl_uniform_batch = (-probs.log().sum(dim=1) / num_classes).sum().item()
        kl_uniform += kl_uniform_batch
        total_samples += inputs.size(0)

    # 求平均
    kl_uniform /= total_samples
    return kl_uniform

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
    GAMMA = 0.97  # 学习率指数衰减系数

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # 模型、损失函数和优化器
    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

    # 初始化 KL_uniform 指标记录
    kl_uniforms = []

    # 训练配置
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        scheduler.step()  # 更新学习率

        # 计算 KL_uniform 指标
        kl_uniform = compute_kl_uniform(model, train_loader, device, max_batches=10)
        kl_uniforms.append(kl_uniform)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, KL_uniform: {kl_uniform:.4f}")

        # 打印训练和测试结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    with open("kl_uniforms_backup.pkl", "wb") as f:
        pickle.dump(kl_uniforms, f)
    print("KL_uniform 数据已备份到 kl_uniforms_backup.pkl")

    # 保存模型
    torch.save(model.state_dict(), "resnet18_cifar10.pth")
    print("训练完成，模型已保存！")

    # 剪枝精度数据
    epochs = [20, 50, 75, 100, 125, 150, 175]
    test_accuracies = [87.70, 89.33, 90.06, 90.05, 90.22, 89.04, 86.05]

    # 绘制图像
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 Test Accuracy 曲线 (左侧 y 轴)
    ax1.plot(range(1, NUM_EPOCHS + 1), [None] * NUM_EPOCHS, color="blue", linestyle="-")  # 全线为实线
    ax1.plot(epochs, test_accuracies, color="blue", linestyle="-", marker="o", label="Test Accuracy (%)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="upper left", fontsize=10)

    # 绘制 KL_uniform 曲线 (右侧 y 轴)
    ax2 = ax1.twinx()
    ax2.plot(range(1, NUM_EPOCHS + 1), kl_uniforms, color="orange", linestyle="-", label="KL_uniform")
    ax2.set_ylabel("KL_uniform", fontsize=12, color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.legend(loc="upper right", fontsize=10)

    # 添加标题
    plt.title("KL_uniform and Test Accuracy during Training", fontsize=14)

    # 添加网格
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 保存和显示图像
    plt.savefig("kl_uniform_test_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()

