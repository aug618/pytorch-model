import torch
import torch.nn as nn


class BasicBlock(nn.Module): #定义一个BasciBlock类，继承nn.Module类，适用于ResNet18和ResNet34
    expansion = 1 #指定膨胀因子为1 ，主分支的卷积核数量等于输入的卷积核数量乘以膨胀因子
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #定义一个卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为3*3，步长为stride，填充为1
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)            
        self.bn1 = nn.BatchNorm2d(out_channels) #定义一个BatchNorm2d层，输入通道数为out_channels
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False) #定义一个ReLU层         
        self.bn2 = nn.BatchNorm2d(out_channels) #定义一个BatchNorm2d层，输入通道数为out_channels
        self.downsample = downsample #定义一个下采样层

    def forward(self, x): #定义前向传播函数
        identity = x #将输入赋值给identity
        out = self.conv1(x) #卷积
        out = self.bn1(out) #BatchNorm2d
        out = self.relu(out) #ReLU
        out = self.conv2(out) #卷积
        out = self.bn2(out) #BatchNorm2d
        if self.downsample is not None: #如果downsample不为空
            identity = self.downsample(x) #下采样
        out += identity #残差连接
        out = self.relu(out) #ReLU
        return out #返回out



class Bottleneck(nn.Module): #定义一个Bottleneck类，适用于ResNet50、ResNet101和ResNet152
    expansion = 4 #指定膨胀因子为4，主分支的卷积核数量等于输入的卷积核数量乘以膨胀因子
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False) #定义一个卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为1
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample         #定义一个下采样层,如果输入输出通道数不一致，需要进行下采样，否则不需要下采样

    def forward(self, x): #定义前向传播函数
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out
    

class ResNet(nn.Module): #定义一个ResNet类
    def __init__(self, block, block_num, num_classes=1000,include_top=True):
        #block为对应网络选取，比如resnet18，34选取的block为BasicBlock，resnet50，101选取的block为Bottleneck
        #block_num 残差结构的数目,比如resnet18，34选取的block_num为[2,2,2,2]，resnet50，101选取的block_num为[3,4,6,3]
        #num_classes 分类数目
        super().__init__()
        self.include_top = include_top #分类头
        self.in_channels = 64
        #定义第一个卷积层，输入通道数为3，输出通道数为64，卷积核大小为7*7，步长为2，填充为3
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_num[0]) #创建四个残差层，分别对应resnet的四个stage
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules(): #初始化权重
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, out_channels, block_num, stride=1): #创建残差层
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride = stride, downsample = downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x): #定义前向传播函数
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x
    

def resnet18(num_classes=1000, include_top=True): #定义resnet18
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)   

def resnet34(num_classes=1000, include_top=True): #定义resnet34
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True): #定义resnet50
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True): #定义resnet101
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)