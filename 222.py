import torch
import torch.nn as nn

# 定义 1x1, 3x3, 1x1 卷积块和残差连接
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(ConvBlock, self).__init__()
        self.downsample = downsample

        # 1x1卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # 3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 1x1卷积
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 下采样层
        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 使用MaxPool进行下采样
        else:
            self.pool = nn.Identity()  # 如果不需要下采样，则不做任何处理

        # 残差连接
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)
        else:
            self.residual = nn.Identity()  # 如果输入和输出通道相同，直接通过

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)

        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)  # 下采样
        out += residual  # 残差连接
        return out

# 构建模型
class CustomNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CustomNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.layer1 = ConvBlock(n_channels, 64, downsample=True)   # 第1层, 输出: [12, 64, 56, 56]
        self.layer2 = ConvBlock(64, 128, downsample=True)           # 第2层, 输出: [12, 128, 28, 28]
        self.layer3 = ConvBlock(128, 320, downsample=True)          # 第3层, 输出: [12, 256, 14, 14]
        self.layer4 = ConvBlock(320, 512, downsample=True)          # 第4层, 输出: [12, 512, 7, 7]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x1 = self.layer1(x)  # 输出尺寸: [12, 64, 56, 56]
        x1=self.pool(x1)
        x2 = self.layer2(x1)  # 输出尺寸: [12, 128, 28, 28]
        x3 = self.layer3(x2)  # 输出尺寸: [12, 256, 14, 14]
        x4 = self.layer4(x3)  # 输出尺寸: [12, 512, 7, 7]
        return x1, x2, x3, x4

# 测试网络
if __name__ == "__main__":
    model = CustomNet(n_channels=3, n_classes=9)
    x = torch.randn(12, 3, 224, 224)  # 输入尺寸: [12, 3, 224, 224]
    outputs = model(x)

    for i, output in enumerate(outputs, 1):
        print(f"Output {i} shape: {output.shape}")
