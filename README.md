
# ModelReproduction

基于 PyTorch 的深度学习模型复现

## 项目介绍

本项目旨在使用 PyTorch 框架复现经典深度学习模型，包括卷积神经网络和其他主流网络架构。每个模型都包含完整的训练和推理代码，并使用 SwanLab 进行实验追踪。

## 参考学习链接

* B站up主霹雳吧啦Wz：[https://github.com/WZMIAOMIAO/deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)
* 李沐大佬的动手学深度学习：[https://courses.d2l.ai/zh-v2/](https://courses.d2l.ai/zh-v2/)

## 环境要求

```
- Python==3.12.7
- torch==2.5.1+cu121
- torchvision==0.20.1
- swanlab==0.4.12
```


## 已实现模型

### 1. LeNet-5 (更新时间: 2025.03.11)

* LeNet-5 由 Yann LeCun 等人在 1998 年提出，用于手写数字识别的卷积神经网络，是卷积神经网络的开山之作。
* 主要用途：手写数字识别。

### 2. AlexNet (更新时间: 2025.03.16)

* AlexNet 由 Alex Krizhevsky 等人在 2012 年提出，在 ImageNet 图像分类竞赛中取得突破性成果，标志着深度学习在计算机视觉领域的崛起。
* 相比 LeNet-5，AlexNet 具有更深的网络结构和更多的参数，首次使用了 ReLU 激活函数、Dropout 和数据增强等技术。
* 主要用途：图像分类。

### 3. VGG16 (更新时间: 2025.03.17)

* VGG16 由牛津大学视觉几何组（Visual Geometry Group）提出，以其简洁而深邃的结构而闻名。
* 其核心特点是使用小尺寸的 3x3 卷积核，通过堆叠多个卷积层来增加网络深度。
* VGG16 在 ImageNet 图像分类挑战赛中表现出色，证明了深度对卷积神经网络性能的重要性。
* 该模型由 16 个带权重的层组成，其中包含 13 个卷积层和 3 个全连接层。
* 虽然 VGG16 在图像分类任务中被广泛使用，但是，现在很多情况下，更小的网络架构会更加的被需要，例如 SqueezeNet，GoogLeNet 等。但是 VGG16 仍然是一个很好的学习对象，因为它很容易被实现。
* 主要用途：图像分类。

### 4. NiN (更新时间: 2025.03.17)

* NiN（Network in Network）引入了"微型神经网络"（micro network）的概念，使用 MLPConv 层代替传统的卷积层，提高了模型的非线性映射能力。
* NiN 的全局平均池化（Global Average Pooling）层也为后续的 GoogLeNet 和 ResNet 等网络提供了重要参考。
* 主要用途：图像分类。

### 5. GoogLeNet (更新时间: 2025.03.18)

* GoogLeNet 引入了 Inception 模块，通过并行使用不同尺寸的卷积核，提高了模型对多尺度特征的提取能力。
* GoogLeNet 的设计目标是提高模型性能的同时，降低计算复杂度。
* 主要用途：图像分类。

### 6. ResNet34 (更新时间: 2025.03.21)

* ResNet34 由微软研究院的何凯明等人在2015年提出，通过引入残差连接（Residual Connection）解决了深层网络的梯度消失问题。
* 残差连接允许网络学习恒等映射，使深层网络至少能够达到与浅层网络相同的性能。
* ResNet34 包含34层，采用基本残差块（Basic Block）构建，相比VGG等网络，参数量更少但性能更优。
* 主要用途：图像分类，也常用作其他计算机视觉任务的骨干网络。

### 7. MobileNetV3 (更新时间: 2025.03.22)

* MobileNetV3 是由Google团队于2019年提出的高效轻量级CNN架构，专为移动设备和边缘计算设计。
* 结合了深度可分离卷积、SE注意力机制和硬swish激活函数，在保持高精度的同时大幅减少参数量和计算量。
* MobileNetV3分为Large和Small两个版本，分别适用于不同的计算资源限制场景。
* 主要用途：移动设备上的图像分类、目标检测和语义分割等任务。

### 8. SwinTransformer (更新时间: 2025.03.23)

* Swin Transformer 由微软研究院在2021年提出，是第一个在计算机视觉主流任务上全面超越CNN的Transformer模型。
* 核心创新是引入了"滑动窗口"（Shifted Window）机制，有效解决了标准Transformer在处理高分辨率图像时的计算复杂度问题。
* 采用层次化结构设计，能够像CNN一样生成多尺度特征图，同时保留了Transformer的全局建模能力。
* 主要用途：图像分类、目标检测、语义分割等多种视觉任务。

## 项目特点

* 注重经典模型的复现与学习。
* 提供完整的代码实现，方便学习者理解和应用。
* 利用 SwanLab 进行实验追踪，便于分析和优化模型。