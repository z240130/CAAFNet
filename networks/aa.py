import torch
import torch.nn as nn
import torch.nn.functional as F


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
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)  # 下采样
        out += residual  # 残差连接
        return out

if __name__ == '__main__':
    conv_block = ConvBlock(in_channels=3, out_channels=16, downsample=True)

    # 创建一个随机输入张量，模拟一个3通道的图像（例如 32x32 图像）
    input_tensor = torch.randn(1, 3, 224, 224)  # batch_size=1，3通道，32x32图像

    # 输出张量
    output_tensor = conv_block(input_tensor)

    # 打印输入和输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
