# Early Pruning Periods Reproduce

paper link：https://openreview.net/pdf?id=tnykAJ5ViF

## 仓库内容

- **代码文件**:
  - `train.py`: 用于在 CIFAR-10 数据集上训练 ResNet18 模型（无剪枝）。
  - `train_with_prune.py`: 在指定的 epoch 对模型进行剪枝并继续训练的脚本。
  - `resnet.py`: 实现 ResNet18 网络结构。
  - `utils.py`: 包含数据预处理和辅助功能的工具函数。

- **数据文件**:
  - `kl_uniforms_backup.pkl`: 训练过程中计算的 KL_uniform 值的备份文件。
  - `resnet18_cifar10.pth`: 在 CIFAR-10 数据集上训练的 ResNet18 模型权重。
  - `resnet18_pruned_cifar10.pth`: 在指定剪枝点训练的 ResNet18 模型权重。

- **可视化文件**:
  - `kl_uniform.png`: 训练过程中 KL_uniform 指标的变化曲线。
  - `kl_uniform_test_accuracy.png`: 显示 KL_uniform 和测试精度的联合图像。
  - `fisher_trace.png`: 与剪枝指标相关的可视化图像。

- **其他文件**:
  - `requirements.txt`: 项目依赖的 Python 包。
  - `.idea/` 和 `__pycache__/`: IDE 配置和 Python 缓存文件。

## 功能特性

1. **剪枝实现**: 实现全局权重剪枝，并可在指定 epoch 进行剪枝。
2. **KL_uniform 指标**: 计算和可视化 KL_uniform 指标，用于衡量模型分布变化。
3. **联合图像可视化**: 绘制 KL_uniform 和测试精度的联合曲线。
4. **预训练模型**: 提供剪枝前后训练的 ResNet18 模型权重文件。

## 安装与使用

### 环境要求
- Python 3.8 或更高版本
- 安装依赖:
  ```bash
  pip install -r requirements.txt

