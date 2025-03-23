# 导入必要的库和模块
import time                                 # 用于测量训练时间
import torch                                # PyTorch深度学习框架
import torch.nn as nn                       # 神经网络模块
import torch.optim as optim                 # 优化器
import torchvision.transforms as transforms # 图像转换和预处理
from torchvision.datasets import CIFAR10    # CIFAR-10数据集
from torch.utils.data import DataLoader     # 数据加载器
import numpy as np                          # 数值计算
from model import resnet34                  # 导入自定义的ResNet34模型
import swanlab                              # 实验跟踪和可视化工具

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)                 # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed_all(seed)        # 设置PyTorch GPU随机种子
    np.random.seed(seed)                    # 设置NumPy随机种子
    torch.backends.cudnn.deterministic = True # 确保cudnn使用确定性算法

set_seed()                                  # 调用函数设置随机种子

# 设备配置：优先使用GPU，如果没有则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")            # 打印使用的设备信息

# 超参数设置
BATCH_SIZE = 128                            # 批处理大小：每次处理的样本数
EPOCHS = 100                                 # 训练轮数：完整遍历数据集的次数
LR = 0.1                                   # 学习率：梯度更新步长
MOMENTUM = 0.9                              # 动量：加速收敛和减轻震荡
WEIGHT_DECAY = 5e-4                         # 权重衰减：L2正则化系数，防止过拟合
NUM_CLASSES = 10                            # 类别数：CIFAR-10有10个类别

# 数据预处理：训练集增强处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # 随机裁剪：添加4像素填充后裁剪到32x32，增加位置鲁棒性
    transforms.RandomHorizontalFlip(),      # 随机水平翻转：增加数据多样性
    transforms.ToTensor(),                  # 转换为Tensor：将图像从[0,255]转换到[0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 标准化：减均值除以标准差
])

# 数据预处理：测试集只需标准化处理，不需要数据增强
transform_test = transforms.Compose([
    transforms.ToTensor(),                  # 转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 使用相同的标准化参数
])

# 加载CIFAR-10数据集
train_dataset = CIFAR10(root='../data/cifar10', train=True, download=False, transform=transform_train)
test_dataset = CIFAR10(root='../data/cifar10', train=False, download=False, transform=transform_test)

# 创建数据加载器：高效加载训练和测试数据
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 训练时打乱数据
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # 测试时不打乱

# CIFAR-10数据集的类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 初始化ResNet34模型
model = resnet34(num_classes=NUM_CLASSES)   # 创建模型实例
model.to(device)                            # 将模型移动到指定设备

# 计算模型参数量
def count_parameters(model):
    """计算模型可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"模型总参数量: {total_params:,}")
print(f"参数量(MB): {total_params * 4 / (1024 * 1024):.2f} MB")  # 假设每个参数是4字节(float32)

# 初始化SwanLab实验追踪：记录训练过程和结果
swanlab.init(
    project="ResNet",                       # 项目名称
    experiment_name="epoch = 100",  # 实验名称
    description="训练ResNet34模型在CIFAR-10数据集上的表现", # 实验描述
    config={                                # 配置参数：记录实验超参数
        "learning_rate": LR,
        "architecture": "ResNet34",
        "dataset": "CIFAR-10",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "parameters": total_params,         # 记录参数量
        "parameters_MB": total_params * 4 / (1024 * 1024)  # 参数量(MB)
    }
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()           # 交叉熵损失函数：适用于多分类问题
# 使用Adam优化器
optimizer = optim.Adam(
    model.parameters(), 
    lr=LR,                                  # 学习率
    betas=(0.9, 0.999),                     # Adam动量参数
    eps=1e-08,                              # 数值稳定性参数
    weight_decay=WEIGHT_DECAY               # 权重衰减参数
)
# 使用ReduceLROnPlateau学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',                             # 监控验证损失最小化
    factor=0.1,                             # 学习率衰减因子
    patience=3,                             # 容忍多少个周期不改善
    verbose=True                            # 打印学习率变化
)

# 计算Top-K准确率的辅助函数
def calculate_topk_accuracy(outputs, targets, topk=(1, 5)):
    """
    计算Top-K准确率
    Args:
        outputs: 模型输出的预测结果
        targets: 真实标签
        topk: 要计算的top-k值
    Returns:
        Top-K准确率列表
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

# 定义单个训练周期的函数
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()                           # 设置模型为训练模式（启用Dropout和BatchNorm）
    running_loss = 0.0                      # 累计损失值
    correct = 0                             # 正确预测的样本数
    total = 0                               # 总样本数
    
    start_time = time.time()                # 记录开始时间
    
    # 遍历数据加载器中的每个批次
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device) # 将数据移动到指定设备
        
        optimizer.zero_grad()               # 清零梯度：防止梯度累积
        
        outputs = model(inputs)             # 前向传播：获取模型预测输出
        loss = criterion(outputs, labels)   # 计算损失：预测值与真实值的差异
        
        loss.backward()                     # 反向传播：计算梯度
        optimizer.step()                    # 参数更新：根据梯度更新模型参数
        
        # 统计当前批次的损失和准确率
        running_loss += loss.item()         # 累加损失值
        _, predicted = outputs.max(1)       # 获取最大概率的类别索引
        total += labels.size(0)             # 累加样本总数
        correct += predicted.eq(labels).sum().item() # 累加正确预测的样本数
        
        # 打印训练进度
        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
            print(f"\r批次 [{i+1}/{len(dataloader)}] 损失: {running_loss/(i+1):.3f} 准确率: {100.*correct/total:.2f}%", end="")
    
    # 计算整个周期的平均损失和准确率
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    epoch_time = time.time() - start_time   # 计算训练耗时
    
    print(f"\n训练完成，耗时 {epoch_time:.2f}秒。损失: {epoch_loss:.3f} 准确率: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc            # 返回损失和准确率，用于记录

# 定义验证函数
def validate(model, dataloader, criterion, device):
    model.eval()                            # 设置模型为评估模式
    running_loss = 0.0                      # 累计损失值
    correct_top1 = 0                        # Top-1正确预测的样本数
    correct_top5 = 0                        # Top-5正确预测的样本数
    total = 0                               # 总样本数
    
    # 初始化每个类别的正确样本数和总样本数
    class_correct = list(0. for i in range(NUM_CLASSES))
    class_total = list(0. for i in range(NUM_CLASSES))
    
    with torch.no_grad():                   # 禁用梯度计算
        for inputs, labels in dataloader:   
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)         # 前向传播
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # 计算Top-1和Top-5准确率
            top1_acc, top5_acc = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
            
            # 计算Top-1准确率
            _, predicted = outputs.max(1)   # 获取最高概率的类别
            total += labels.size(0)
            correct_top1 += predicted.eq(labels).sum().item()
            
            # 计算Top-5准确率 (对于CIFAR-10，只有10个类别，所以最多取min(5, 10))
            _, pred_top5 = outputs.topk(min(5, NUM_CLASSES), 1, largest=True, sorted=True)
            correct_top5 += torch.eq(pred_top5, labels.view(-1, 1).expand_as(pred_top5)).sum().item()
            
            # 计算每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 计算损失和准确率
    val_loss = running_loss / len(dataloader)
    val_acc_top1 = 100. * correct_top1 / total   # Top-1准确率
    val_acc_top5 = 100. * correct_top5 / total   # Top-5准确率
    
    # 计算错误率
    val_err_top1 = 100. - val_acc_top1    # Top-1错误率
    val_err_top5 = 100. - val_acc_top5    # Top-5错误率
    
    print(f"验证 - 损失: {val_loss:.3f}")
    print(f"Top-1 准确率: {val_acc_top1:.2f}% (错误率: {val_err_top1:.2f}%)")
    print(f"Top-5 准确率: {val_acc_top5:.2f}% (错误率: {val_err_top5:.2f}%)")
    
    # 计算并打印每个类别的准确率
    class_acc = {}
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:              
            acc = 100 * class_correct[i] / class_total[i]
            print(f'{classes[i]} 类别准确率: {acc:.2f}%')
            class_acc[classes[i]] = acc
    
    return val_loss, val_acc_top1, val_acc_top5, val_err_top1, val_err_top5, class_acc

if __name__ == "__main__":
    # 定义保存最佳模型的路径
    save_path = './resnet34_cifar10.pth'
    best_acc = 0.0                             # 初始化最佳准确率
    
    # 主训练循环：遍历每个训练周期
    for epoch in range(EPOCHS):
        print(f"\n第 {epoch+1}/{EPOCHS} 轮")   # 打印当前周期
        print("-" * 30)                        # 分隔线
        
        # 训练一个周期
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 在验证集上评估模型性能
        val_loss, val_acc_top1, val_acc_top5, val_err_top1, val_err_top5, class_acc = validate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)                       # 根据验证损失调整学习率
        current_lr = optimizer.param_groups[0]['lr']   # 获取当前学习率
        
        # 将指标记录到SwanLab
        metrics = {
            "train/loss": train_loss,          # 训练损失
            "train/accuracy": train_acc,       # 训练准确率
            "val/loss": val_loss,              # 验证损失
            "val/accuracy_top1": val_acc_top1, # Top-1验证准确率
            "val/accuracy_top5": val_acc_top5, # Top-5验证准确率
            "val/error_top1": val_err_top1,    # Top-1错误率
            "val/error_top5": val_err_top5,    # Top-5错误率
            "learning_rate": current_lr        # 当前学习率
        }
        
        # 添加每个类别的准确率到指标字典
        for cls, acc in class_acc.items():
            metrics[f"val/accuracy_{cls}"] = acc
        
        swanlab.log(metrics)                   # 记录所有指标
        
        # 保存性能最好的模型 (基于Top-1准确率)
        if val_acc_top1 > best_acc:            # 如果当前验证准确率更高
            best_acc = val_acc_top1            # 更新最佳准确率
            torch.save(model.state_dict(), save_path) # 保存模型参数
            print(f"保存最佳模型，Top-1准确率: {best_acc:.2f}%")
    
    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%") # 打印训练结束信息
    
    # 加载最佳模型并进行最终评估
    model.load_state_dict(torch.load(save_path)) # 加载最佳模型参数
    final_loss, final_acc_top1, final_acc_top5, final_err_top1, final_err_top5, final_class_acc = validate(model, test_loader, criterion, device) # 在测试集上进行评估
    
    print(f"\n最终测试结果:")
    print(f"  - Top-1 准确率: {final_acc_top1:.2f}% (错误率: {final_err_top1:.2f}%)")
    print(f"  - Top-5 准确率: {final_acc_top5:.2f}% (错误率: {final_err_top5:.2f}%)")
    print(f"  - 模型参数量: {total_params:,} ({total_params * 4 / (1024 * 1024):.2f} MB)")
    
