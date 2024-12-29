import torch
import torch.nn as nn
import torch.optim as optim
from resnet import ResNet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

def check_model_sparsity(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += torch.sum(param.abs() < 1e-8).item()
    sparsity = zero_params / total_params * 100
    print(f"Total Parameters: {total_params}")
    print(f"Zero Parameters: {zero_params}")
    print(f"Sparsity: {sparsity:.2f}%")

def test_model(model, test_loader, criterion, device):
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

def plot_weight_distribution(model):
    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights += list(param.data.cpu().numpy().flatten())
    zero_weights = [w for w in all_weights if abs(w) < 1e-8]
    non_zero_weights = [w for w in all_weights if abs(w) >= 1e-8]
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_weights, bins=100, color="blue", alpha=0.7, label="Non-Zero Weights")
    plt.hist(zero_weights, bins=100, color="red", alpha=0.5, label="Zero Weights")
    plt.title("Weight Distribution After Pruning")
    plt.xlabel("Weight Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(num_classes=10).to(device)
    model.load_state_dict(torch.load("resnet18_pruned_cifar10.pth"), strict=False)
    model.eval()
    print("剪枝后的模型已成功加载！")
    print("检查模型稀疏性...")
    check_model_sparsity(model)
    print("绘制权重分布直方图...")
    plot_weight_distribution(model)
    test_loader = get_dataloaders()[1]
    criterion = torch.nn.CrossEntropyLoss()
    print("测试剪枝后模型的性能...")
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    baseline_acc = 95.0
    if test_acc >= baseline_acc - 1.0:
        print(f"Pruning successful: Test Accuracy {test_acc:.2f}% meets baseline.")
    else:
        print(f"Pruning unsuccessful: Test Accuracy {test_acc:.2f}% below baseline.")

if __name__ == "__main__":
    main()
