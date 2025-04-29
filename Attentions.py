import torch
import os
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import math

# CA Attention
class Category_attention(nn.Module):
    # Input dimension, num of classes, k: the number of channels allocated to each class
    def __init__(
            self,
            inputs,
            num_classes,
            k=5):
        super(Category_attention, self).__init__()
        self.num_class = num_classes
        self.k = k
        self.conv1 = nn.Conv2d(inputs, num_classes * k, 1, padding='same')
        self.bn1 = nn.BatchNorm2d(num_classes * k)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, input):
        n, c, h, w = input.size()
        F1 = self.conv1(input)
        F1 = self.bn1(F1)
        F1 = self.relu(F1)

        F2 = self.maxpool(F1)
        F2 = F2.view(n, self.num_class, self.k)
        F3 = F2.mean(dim=-1)

        F4 = F1.view(n, self.num_class, h, w, self.k)
        F4 = F4.mean(dim=-1)

        F5 = F4 * F3.view(n, self.num_class, 1, 1)
        M = F5.mean(dim=1)

        out = input * M

        return out

# ECA Attention
class EfficientChannelAttention(nn.Module):  # Efficient Channel Attention module
    def __init__(self, channel, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        # t = int(abs((math.log(c, 2) + b) / gamma))
        k = self.adaptive_kernel_size(channel, b, gamma)
        print(k)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def adaptive_kernel_size(self, channel, b, gamma):
        """Calculate adaptive kernel size based on input channel number."""
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1  # Ensure k_size is odd
        return k

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# CAM Attention
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
    # Residual Connection Benchmark
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        out = identity * x_w * x_h

        return out
