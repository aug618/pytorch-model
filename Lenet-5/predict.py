# 导入必要的库
import torch  # 导入PyTorch库
import torchvision.transforms as transforms  # 导入图像预处理工具
from model import LeNet  # 从model.py文件导入LeNet模型
from PIL import Image  # 导入图像处理库

# 检查CUDA是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义图像预处理流程：调整图像大小为32x32，转换为张量，标准化像素值
transform = transforms.Compose([transforms.Resize((32, 32)),
    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 定义10个类别的名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 创建LeNet模型实例
net = LeNet()
# 加载预训练模型参数
net.load_state_dict(torch.load('./lenet.pth', map_location=device,weights_only=True))
# 将模型移动到指定设备
net.to(device)
# 设置为评估模式
net.eval()

# 打开测试图像
img = Image.open('images/2.jpg')
# 对图像进行预处理
img = transform(img)
# 增加一个批次维度，因为模型期望输入是[batch_size, channels, height, width]格式
img = img.unsqueeze(0)  # 修正了错误
# 将图像移动到指定设备
img = img.to(device)

# 禁用梯度计算，因为我们只是做推理
with torch.no_grad():
    # 将图像输入到模型中获取输出
    outputs = net(img)
    # 获取最大概率的类别索引
    predict_y = torch.max(outputs, dim=1)[1].cpu().numpy()
# 打印预测的类别名称
print(classes[predict_y[0]])
