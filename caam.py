import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, relu=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels)
    ]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, stride=1, padding=1, relu=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class CAAM(nn.Module):
    def __init__(self, in_channel, channel, num_classes=2):
        super(CAAM, self).__init__()
        self.channel = channel
        self.num_classes = num_classes

        # 查询、键、值卷积层
        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                       conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                         conv(channel, channel, 1, relu=True))
        self.conv_key1= nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        # 输出卷积层
        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = nn.Conv2d(channel, num_classes, 1)  # 输出 num_classes 类别


    def forward(self, x, map,polt):
        b, c, h, w = x.shape

        # 确保 map 的分辨率与输入特征一致
        map = F.interpolate(map, size=(h, w), mode='bilinear', align_corners=False)


        map_expanded = map.unsqueeze(2)  # Shape: (B, C, 1, H, W)
        diff = torch.abs(map_expanded - map.unsqueeze(1))  # Pairwise differences, Shape: (B, C, C, H, W)
        confusion_area = torch.max(diff, dim=2).values  # Take maximum across class pairs
        confusion_area =torch.clamp(confusion_area, 0, 1)
        # Reshape inputs for context computation
        f = x.reshape(b, h * w, -1).contiguous()  # Flatten spatial dimensions
        confusion_area = confusion_area.reshape(b, self.num_classes, h * w).contiguous()

        # 计算上下文向量
        context = torch.bmm(confusion_area, f).permute(0, 2, 1).unsqueeze(3)
        # 对x的qkv
        # q = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        # k = self.conv_key1(x).view(b, self.channel, -1)
        # sim1 = torch.bmm(q, k)
        # b, hw1, hw2 = sim1.shape  # 获取 batch size 和注意力图的宽高
        # size = (224, 224)  # 目标大小
        # sim_unsqueezed = sim1.unsqueeze(1)
        #将相似度矩阵调整大小
        # if polt:
        #
        #
        #     import matplotlib.pyplot as plt
        #     import seaborn as sns
        #     sim_resized = F.interpolate(sim_unsqueezed, size=size, mode='bilinear', align_corners=False)
        #     # 可视化第一张图
        #     sim_resized = sim_resized.squeeze(1)
        #     sns.heatmap(sim_resized[0].cpu().detach().numpy(), cmap='viridis')
        #     plt.show()
        # 查询、键、值计算
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        sim = torch.bmm(query, key)

        eps = 1e-8
        sim = ((self.channel ** -0.5) * sim) + eps
        sim = F.softmax(sim, dim=-1)

        # 计算精细特征
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)

        context = self.conv_out1(context)

        # 特征融合
        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)  # 输出 logits
#

        return x, out

def _save_attn_map(self, sim, step, sample_idx, class_idx):
    """
    sim: (B, H*W, num_classes)
    """
    B, HW, C = sim.shape
    H = W = int(HW**0.5)
    attn = sim[sample_idx, :, class_idx].view(H, W).detach().cpu().numpy()
    plt.figure(figsize=(4,4))
    plt.imshow(attn)
    plt.axis('off')
    plt.title(f'step{step}_s{sample_idx}_c{class_idx}')
    plt.savefig(os.path.join(self.save_dir, f'attn_step{step:04d}.png'),
                bbox_inches='tight')
    plt.close()





