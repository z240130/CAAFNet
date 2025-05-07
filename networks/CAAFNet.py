import os

import numpy as np
import timm
import torch
import torch.nn as nn

from networks.decoder111 import MultiStageDecoder
from networks.segformer import *

from typing import Tuple
from einops import rearrange


from caam import CAAM, conv

num_classes=1









class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out



class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out




class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=False)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class ECAModule(nn.Module):
    def __init__(self, kernel_size=3):
        super(ECAModule, self).__init__()
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.size()
        # Global Average Pooling (GAP)
        y = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        y = y.view(B, C, 1).permute(0, 2, 1)  # Reshape to [B, C, 1]
        # 1D Convolution

        y = self.conv1d(y)  # [B, C, 1]
        y = self.sigmoid(y)  # Apply Sigmoid
        y = y.view(B, C, 1, 1)  # Reshape back to [B, C, 1, 1]
        # Channel-wise multiplication
        return x * y
class MMFIB(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(MMFIB, self).__init__()

        # channel attention for F_g, use SE Block
        # self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        # self.relu = nn.ReLU(inplace=False)
        # self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=False)
        # ECA for Transformer branch
        self.eca_x = ECAModule(kernel_size=3)
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # # channel attetion for transformer branch
        # x_in = x
        # x = x.mean((2, 3), keepdim=True)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.sigmoid(x) * x_in
        # ECA for Transformer branch
        x = self.eca_x(x)  # Apply ECA to Transformer branch
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1082'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1082'


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

class CAAFNet(nn.Module):
    #torch.autograd.set_detect_anomaly(True)
    def __init__(self, num_classes=num_classes, token_mlp_mode="mix_skip", encoder_pretrained=True, drop_rate=0.2):
        super().__init__()

        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        d_base_feat_size = 7  # for 224 input size
        in_out_chan = [[32, 64], [144, 128], [288, 320], [512, 512]]

        dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]
        self.backbone = MiT(224, dims, layers, token_mlp_mode)  # Transformer encoder
        self.backbone2 = CustomNet(n_channels=3, n_classes=num_classes)  # CNN encoder

        self.reduction_ratios = [1, 2, 4, 8]

        self.MMFIB1=MMFIB(ch_1=64,ch_2=64,r_2=4, ch_int=64, ch_out=64, drop_rate=drop_rate / 2)
        self.MMFIB2 = MMFIB(ch_1=128, ch_2=128, r_2=4, ch_int=128, ch_out=128, drop_rate=drop_rate / 2)
        self.MMFIB3 = MMFIB(ch_1=320, ch_2=320, r_2=4, ch_int=320, ch_out=320, drop_rate=drop_rate / 2)
        self.MMFIB4 = MMFIB(ch_1=512, ch_2=512, r_2=4, ch_int=512, ch_out=512, drop_rate=drop_rate / 2)


        # 解码器模块
        # self.decoder = PAA_d(64)
        self.CAAM4= CAAM(576,128,num_classes)
        self.CAAM3=  CAAM(448, dims[1],num_classes)
        self.CAAM2 = CAAM(256 , dims[0], num_classes)
        self.CAAM1 = CAAM(128, dims[0], num_classes)
        #self.conv=nn.Conv2d(in_channels=64,out_channels=9,kernel_size=(1,1))
        self.decoder=MultiStageDecoder([64, 128, 320, 512],[64,64, 128, 320],num_classes)
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.con=nn.Conv2d(512,320,1)

        self.pvt = timm.create_model(
            'pvt_v2_b5',
            pretrained=True,
            features_only=True,

        )
        self.transformer4 = timm.create_model("vit_base_r50_s16_224.orig_in21k", pretrained=True)
        self.patch_embed = self.transformer4.patch_embed.proj
        self.transformer4 = self.transformer4.blocks
        self.conv_more1 = Conv2dReLU(
            768,
            512,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

    def tr_reshape(self, x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        return x
    def forward(self, x):


        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        base_size = x.shape[-2:]
        # 提取Transformer和CNN编码器的多层特征
        encoder = self.pvt(x)  # Transformer encoder
        encoder2 = self.backbone2(x)  # CNN encoder

        e1, e2, e3, e4 = encoder[0], encoder[1], encoder[2], encoder[3]
        r1, r2, r3, r4 = encoder2[0], encoder2[1], encoder2[2], encoder2[3]

        # 层级特征融合
        e11 = self.MMFIB1(r1, e1)
        e22 = self.MMFIB2(r2, e2)
        e33 = self.MMFIB3(r3, e3)
        #e44 = self.ronghe4(r4, e4)
        e44 = torch.cat((e4,r4),dim=1)
        #111 vit
        # tr4
        e44 = self.patch_embed(e44)
        x41 = e44.flatten(2).transpose(1, 2)
        tr4 = self.transformer4(x41)
        tr4 = self.tr_reshape(tr4)
        x5 = self.conv_more1(tr4)



        features=[x5,e33,e22,e11]

        f5, a5 = self.decoder(features)
        # f5, a5 = self.decoder(x4_o, x3_o, x2_o,x1_o)

        out5 = self.res(a5, base_size)
        f4,a4=self.CAAM4(torch.cat([x5,self.ret(f5,x5)],dim=1),a5,False)
        #out4 = self.res(a4, base_size)
        f3,a3=self.CAAM3(torch.cat([e33,self.ret(f4,e33)],dim=1),a4,False)
        #out3 = self.res(a3, base_size)
        f2, a2= self.CAAM2(torch.cat([e22, self.ret(f3, e22)], dim=1), a3,False)
        #out2 = self.res(a2, base_size)
        _,a1=self.CAAM1(torch.cat([e11, self.ret(f2, e11)], dim=1), a2,True)
        out1 = self.res(a1, base_size)

        return out1

