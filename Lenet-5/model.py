import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms 

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  # 初始化父类nn.Module
        
        # 第一个卷积层: 输入3通道(RGB)，输出16特征图，卷积核大小5x5
        # 输入为 32x32x3 的图像，经过卷积后变为 28x28x16
        self.conv1 = nn.Conv2d(3, 16, 5)
        
        # 第一个最大池化层: 窗口大小2x2，步长2
        # 将特征图从 28x28x16 缩小为 14x14x16
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 第二个卷积层: 输入16特征图，输出32特征图，卷积核大小5x5
        # 输入为 14x14x16，经过卷积后变为 10x10x32
        self.conv2 = nn.Conv2d(16, 32, 5)
        
        # 第二个最大池化层: 窗口大小2x2，步长2
        # 将特征图从 10x10x32 缩小为 5x5x32
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 第一个全连接层: 将展平的特征图(5x5x32=800个神经元)连接到120个神经元
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        
        # 第二个全连接层: 120个神经元连接到84个神经元
        self.fc2 = nn.Linear(120, 84)
        
        # 输出层: 84个神经元连接到10个神经元(CIFAR10的10个类别)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入图像经过第一个卷积层，再通过ReLU激活函数，再经过最大池化
        x = self.pool1(F.relu(self.conv1(x)))
        
        # 第一层池化的输出经过第二个卷积层，再通过ReLU激活函数，再经过最大池化
        x = self.pool2(F.relu(self.conv2(x)))
        
        # 将卷积层输出的特征图展平为一维张量，-1表示自动计算批次大小
        x = x.view(-1, 32 * 5 * 5)
        
        # 展平后的特征通过第一个全连接层和ReLU激活函数
        x = F.relu(self.fc1(x))
        
        # 第一个全连接层的输出通过第二个全连接层和ReLU激活函数
        x = F.relu(self.fc2(x))
        
        # 第二个全连接层的输出通过输出层(不使用激活函数，因为后续会使用交叉熵损失)
        x = self.fc3(x)
        
        # 返回模型输出(logits)，用于计算损失和预测类别
        return x