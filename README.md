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

### 2. AlexNet (更新时间: 2025.03.16)

* AlexNet 由 Alex Krizhevsky 等人在 2012 年提出，在 ImageNet 图像分类竞赛中取得突破性成果，标志着深度学习在计算机视觉领域的崛起。
* 相比 LeNet-5，AlexNet 具有更深的网络结构和更多的参数，首次使用了 ReLU 激活函数、Dropout 和数据增强等技术。

### 3.VGG16 (更新时间: 2025.03.17)
* VGG16 由牛津大学视觉几何组（Visual Geometry Group）提出，以其简洁而深邃的结构而闻名。
其核心特点是使用小尺寸的 3x3 卷积核，通过堆叠多个卷积层来增加网络深度。
* VGG16 在 ImageNet 图像分类挑战赛中表现出色，证明了深度对卷积神经网络性能的重要性。
* 该模型由16个带权重的层组成，其中包含13个卷积层和3个全连接层。
虽然VGG16在图像分类任务中被广泛使用，但是，现在很多情况下，更小的网络架构会更加的被需要，例如SqueezeNet，GoogLeNet等。但是VGG16仍然是一个很好的学习对象，因为它很容易被实现。
## 项目特点

* 注重经典模型的复现与学习。
* 提供完整的代码实现，方便学习者理解和应用。
* 利用 SwanLab 进行实验追踪，便于分析和优化模型。

## 补充说明

* 项目利用了swanlab，这是一个良好的习惯，可以方便的记录实验的各项参数，方便实验的对比，和调优。
* 希望该项目可以持续更新更多的经典的神经网络模型。
* 在实际操作中，请确保您的环境配置与项目要求一致，以避免潜在的兼容性问题。