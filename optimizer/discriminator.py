import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np


class PatchGANDiscriminator(nn.Module):
    """PatchGAN判别器 - 判断局部patch的真实性"""

    def __init__(self, input_channels=3, ndf=64, n_layers=3):
        super().__init__()

        # 使用谱归一化提高训练稳定性
        layers = []

        # 第一层：不使用BN
        layers.append(spectral_norm(nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 中间层
        in_channels = ndf
        for i in range(n_layers):
            out_channels = min(ndf * (2 ** (i + 1)), 512)
            layers.append(spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        # 输出层：1个通道表示每个patch的真实性
        layers.append(spectral_norm(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

        # 特征提取器用于特征匹配损失
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x, return_features=False):
        """前向传播"""
        features = self.feature_extractor(x)
        output = self.model[-2:](features)

        if return_features:
            return output, features
        return output  # 返回单个张量，不是列表


class MultiScaleDiscriminator(nn.Module):
    """多尺度判别器 - 在不同分辨率下判断"""

    def __init__(self, input_channels=3):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(input_channels, ndf=64, n_layers=3),  # 原尺度
            PatchGANDiscriminator(input_channels, ndf=64, n_layers=2),  # 1/2尺度
            PatchGANDiscriminator(input_channels, ndf=64, n_layers=1),  # 1/4尺度
        ])

    def forward(self, x):
        """返回多个尺度的判别结果（列表）"""
        outputs = []

        # 原尺度
        outputs.append(self.discriminators[0](x))

        # 下采样后的尺度
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        outputs.append(self.discriminators[1](x_half))

        x_quarter = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        outputs.append(self.discriminators[2](x_quarter))

        return outputs  # 返回列表，每个元素是张量