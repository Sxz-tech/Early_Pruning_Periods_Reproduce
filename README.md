# Early Pruning Periods Reproduce

paper link：https://openreview.net/pdf?id=tnykAJ5ViF

## 仓库内容

**代码文件**:
`train.py`: 用于在 CIFAR-10 数据集上训练 ResNet18 模型，并且绘制KLuniform图线以及在不同epoch处剪枝得到的精度曲线图像。
`train_with_prune.py`: 在指定的 epoch 处对模型进行90%剪枝，然后以论文中规定的方法更新学习率后继续训练该模型，并且测得最终模型准确率。
`resnet.py`: 按照论文参数要求实现 ResNet18 网络结构。
`utils.py`: 测试得到的模型稀疏度的工具函数，即判断是否剪枝成功。

**数据文件**:
`kl_uniforms_backup.pkl`: 训练过程中计算的 KL_uniform 值的备份文件。
`resnet18_cifar10.pth`: 在 CIFAR-10 数据集上训练的 ResNet18 模型权重。
`resnet18_pruned_cifar10.pth`: 在指定剪枝点训练的 ResNet18 模型权重。

**可视化文件**:
`kl_uniform.png`: 训练过程中 KL_uniform 指标的变化曲线。
`kl_uniform_test_accuracy.png`: 参照论文生成的KL_uniform和不同剪枝模型准确率变化曲线的联合图像。
`fisher_trace.png`: 训练过程中Fisher迹的可视化图像。




