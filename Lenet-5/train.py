# 导入PyTorch库
import torch
# 导入PyTorch视觉库
import torchvision
# 导入图像变换工具
import torchvision.transforms as transforms
# 导入自定义的LeNet模型
from model import LeNet
# 导入神经网络模块
import torch.nn as nn

import swanlab

# 检查是否有可用的CUDA设备，如果有则使用GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 打印使用的设备信息
print(f"Using {device} device")

# 定义数据预处理步骤：转换为Tensor并进行标准化（使均值为0，标准差为1）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载CIFAR10训练数据集（包含50000张图片）
trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform)

# 创建训练数据加载器，设置批量大小为128，打乱数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

# 加载CIFAR10测试数据集（包含10000张图片）
testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform)

# 创建测试数据加载器，批量大小为10000（全部测试集）
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)

# 创建测试数据迭代器
test_data_iter = iter(testloader)
# 获取一批测试数据和标签
test_images, test_labels = next(test_data_iter)
# 将测试数据和标签移到指定设备上（CPU或GPU）
test_images, test_labels = test_images.to(device), test_labels.to(device)

# 实例化LeNet模型并移到指定设备上
net = LeNet().to(device)
# 定义交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义Adam优化器，学习率为0.001
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)





swanlab.init(
    # 设置项目名
    project="Lenet-5",
    
    # 设置超参数
    config={
        "learning_rate": 0.001,
        "architecture": "CNN",
        "dataset": "CIFAR-10",
        "epochs": 10
    }
)





# 开始训练，共5轮
for epoch in range(10):
    # 初始化累计损失为0
    running_loss = 0.0
    # 遍历训练数据批次
    for i, data in enumerate(trainloader, start=0):
        # 获取输入数据和标签
        inputs, labels = data
        # 将数据和标签移到指定设备上
        inputs, labels = inputs.to(device), labels.to(device)

        # 清除梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = net(inputs)

        # 计算损失
        loss = loss_fn(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累加损失
        running_loss += loss.item()
        # 每300批次评估一次模型性能
        if i % 300 == 299:
            # 关闭梯度计算
            with torch.no_grad():
                # 对测试集进行预测
                outputs = net(test_images)
                # 获取最大概率的预测类别
                predict_y = torch.max(outputs, dim=1)[1]
                # 计算准确率
                accuracy = torch.eq(predict_y, test_labels).sum().item() / test_labels.size(0)
                  # 记录训练指标
                swanlab.log({"acc": accuracy, "loss": running_loss/300})
                # 打印训练信息：轮次、步数、平均损失、测试准确率
                print(f'epoch: {epoch}, step: {i} loss: {running_loss / 300:.3f} test accuracy: {accuracy:.3f}')
                # swanlab记录训练指标
                swanlab.log({"acc": accuracy, "loss": running_loss/300})
                # 重置累计损失
                running_loss = 0.0

# 打印训练完成信息
print('Finished Training')
# 保存模型参数
torch.save(net.state_dict(), './lenet.pth')