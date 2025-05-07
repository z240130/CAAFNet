import numpy as np
import torch
from torch import nn
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, skip=None):

        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)

        x = self.conv2(x)

        return x


class MultiStageDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_classes=4, use_batchnorm=True):
        super().__init__()
        self.decoder4 = DecoderBlock(encoder_channels[3], decoder_channels[3], skip_channels=320, use_batchnorm=use_batchnorm)
        self.decoder3 = DecoderBlock(decoder_channels[3], decoder_channels[2], skip_channels=128, use_batchnorm=use_batchnorm)
        self.decoder2 = DecoderBlock(decoder_channels[2], decoder_channels[1], skip_channels=64, use_batchnorm=use_batchnorm)
        self.decoder1 = DecoderBlock(decoder_channels[1], decoder_channels[0], skip_channels=encoder_channels[0], use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)

    def forward(self, features):
        # features: [e44, e33, e22, e11]
        e44, e33, e22, e11 = features

        d4 = self.decoder4(e44,e33)  # 第四阶段解码，跳跃连接 e44
        d3 = self.decoder3(d4, e22)  # 第三阶段解码，跳跃连接 e33
        d2 = self.decoder2(d3, e11)  # 第二阶段解码，跳跃连接 e22
        out = self.final_conv(d2)  # 最终输出
        return d2,out