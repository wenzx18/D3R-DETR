"""
D3R-DETR: DETR with Dual-Domain Density Refinement for Tiny Object Detection in Aerial Images
Copyright (c) 2026 The D3R-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from SFS-Conv (https://github.com/like413/SFS-Conv)
Copyright (c) 2024 The SFS-Conv Authors. All Rights Reserved.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        c = self.conv(x)
        c = self.bn(c)
        c = self.act(c)
        return c


class FractionalGaborFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order, angles, scales):
        super(FractionalGaborFilter, self).__init__()

        self.real_weights = nn.ParameterList()
        self.imag_weights = nn.ParameterList()

        for angle in angles:
            for scale in scales:
                real_weight = self.generate_fractional_gabor(in_channels, out_channels, kernel_size, order, angle, scale)
                self.real_weights.append(nn.Parameter(real_weight))

    def generate_fractional_gabor(
        self, in_channels, out_channels, size, order, angle, scale):

        x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
        x_theta = x * np.cos(angle) + y * np.sin(angle)
        y_theta = -x * np.sin(angle) + y * np.cos(angle)

        real_part = np.exp(-((x_theta**2 + (y_theta / scale) ** 2) ** order)) * np.cos(2 * np.pi * x_theta / scale)
        # Reshape to match the specified out_channels and size
        real_weight = torch.tensor(real_part, dtype=torch.float32).view(1, 1, size[0], size[1])
        # Repeat along the out_channels dimension
        real_weight = real_weight.repeat(out_channels, 1, 1, 1)

        return real_weight

    def forward(self, x):
        real_weights = [weight for weight in self.real_weights]

        real_result = sum(weight * x for weight in real_weights)

        return real_result


class GaborSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order, angles, scales):
        super(GaborSingle, self).__init__()
        self.gabor = FractionalGaborFilter(in_channels, out_channels, kernel_size, order, angles, scales)
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.gabor(self.t)
        out = F.conv2d(x, out, stride=1, padding=(out.shape[-2] - 1) // 2)
        out = self.relu(out)
        out = F.dropout(out, 0.3)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)  # Padding on the left and top
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class GaborFPU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        order=0.25,
        angles=[0, 45, 90, 135],
        scales=[1, 2, 3, 4],
    ):
        super(GaborFPU, self).__init__()
        self.gabor = GaborSingle(in_channels // 4, out_channels // 4, (3, 3), order, angles, scales)
        self.fc = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        x_out = torch.cat(
            [self.gabor(x1), self.gabor(x2), self.gabor(x3), self.gabor(x4)], dim=1
        )
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class FrFTFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super(FrFTFilter, self).__init__()

        self.register_buffer(
            "weight",
            self.generate_FrFT_filter(in_channels, out_channels, kernel_size, f, order),
        )

    def generate_FrFT_filter(self, in_channels, out_channels, kernel, f, p):
        N = out_channels
        d_x = kernel[0]
        d_y = kernel[1]
        x = np.linspace(1, d_x, d_x)
        y = np.linspace(1, d_y, d_y)
        [X, Y] = np.meshgrid(x, y)

        real_FrFT_filterX = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filterY = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filter = np.zeros([d_x, d_y, out_channels])
        for i in range(N):
            real_FrFT_filterX[:, :, i] = np.cos(-f * (X) / math.sin(p) + (f * f + X * X) / (2 * math.tan(p)))
            real_FrFT_filterY[:, :, i] = np.cos(-f * (Y) / math.sin(p) + (f * f + Y * Y) / (2 * math.tan(p)))
            real_FrFT_filter[:, :, i] = (real_FrFT_filterY[:, :, i] * real_FrFT_filterX[:, :, i])
        g_f = np.zeros((kernel[0], kernel[1], in_channels, out_channels))
        for i in range(N):
            g_f[:, :, :, i] = np.repeat(real_FrFT_filter[:, :, i : i + 1], in_channels, axis=2)
        g_f = np.array(g_f)
        g_f_real = g_f.reshape((out_channels, in_channels, kernel[0], kernel[1]))

        return torch.tensor(g_f_real).type(torch.FloatTensor)

    def forward(self, x):
        x = x * self.weight
        return x

    def generate_FrFT_list(self, in_channels, out_channels, kernel, f_list, p):
        FrFT_list = []
        for f in f_list:
            FrFT_list.append(self.generate_FrFT_filter(in_channels, out_channels, kernel, f, p))
        return FrFT_list


class FrFTSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super().__init__()
        self.fft = FrFTFilter(in_channels, out_channels, kernel_size, f, order)
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fft(self.t)
        out = F.conv2d(x, out, stride=1, padding=(out.shape[-2] - 1) // 2)
        out = self.relu(out)
        out = F.dropout(out, 0.3)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class FourierFPU(nn.Module):
    def __init__(self, in_channels, out_channels, order=0.25):
        super().__init__()
        self.fft1 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.25, order)
        self.fft2 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.50, order)
        self.fft3 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.75, order)
        self.fft4 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 1.00, order)
        self.fc = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        x_out = torch.cat([self.fft1(x1), self.fft2(x2), self.fft3(x3), self.fft4(x4)], dim=1)
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class FractionalHaarFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, direction, order):
        super().__init__()
        self.kernel_size = kernel_size
        self.direction = direction
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.order = order
        
        self.register_buffer('mask', self.generate_fractional_mask())

    def generate_fractional_mask(self):
        k = self.kernel_size
        if isinstance(k, int):
            kx, ky = k, k
        else:
            kx, ky = k

        t = torch.linspace(0, 1, kx)

        # left: (0 <= t < 0.5): t^order
        # right: (0.5 <= t <= 1): t^order - 2 * (t - 0.5)^order
        half_idx = kx // 2        
        haar_1d = torch.zeros(kx)
        eps = 1e-6

        t_left = t[:half_idx]
        val_left = torch.pow(t_left + eps, self.order)
        haar_1d[:half_idx] = val_left
        t_right = t[half_idx:]
        val_right = torch.pow(t_right + eps, self.order) - 2 * torch.pow((t_right - 0.5).clamp(min=0) + eps, self.order)
        haar_1d[half_idx:] = val_right
        haar_1d = haar_1d / (haar_1d.abs().max() + eps)

        if self.order < 0.05:
             haar_1d[:half_idx] = -1
             haar_1d[half_idx:] = 1

        mask = torch.zeros((kx, ky))
        if self.direction == 'h':
            mask = haar_1d.view(1, -1).repeat(kx, 1)
        elif self.direction == 'v':
            mask = haar_1d.view(-1, 1).repeat(1, ky)
            
        # [out, in, k, k]
        mask = mask.view(1, 1, kx, ky)
        mask = mask.repeat(self.out_channels, self.in_channels, 1, 1)
        
        return mask

    def forward(self, t):
        return t * self.mask


class HaarSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), direction='h', dilation=1, order=0.5):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.haar_filter = FractionalHaarFilter(in_channels, out_channels, kernel_size, direction, order)
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        weight = self.haar_filter(self.t)        
        k_h, k_w = self.kernel_size
        pad_h = (self.dilation * (k_h - 1)) // 2
        pad_w = (self.dilation * (k_w - 1)) // 2
        out = F.conv2d(x, weight, stride=1, padding=(pad_h, pad_w), dilation=self.dilation)
        out = self.relu(out)
        out = F.dropout(out, 0.3)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class HaarFPU(nn.Module):
    def __init__(self, in_channels, out_channels, order=0.25):
        super(HaarFPU, self).__init__()

        c_in = in_channels // 4
        c_out = out_channels // 4

        # Group 1
        self.haar1 = HaarSingle(c_in, c_out, (3,3), 'h', dilation=1, order=order)
        # Group 2
        self.haar2 = HaarSingle(c_in, c_out, (3,3), 'v', dilation=1, order=order)
        # Group 3
        self.haar3 = HaarSingle(c_in, c_out, (3,3), 'h', dilation=2, order=order)
        # Group 4
        self.haar4 = HaarSingle(c_in, c_out, (3,3), 'v', dilation=2, order=order)
        self.fc = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        out = torch.cat([
            self.haar1(x1),
            self.haar2(x2),
            self.haar3(x3),
            self.haar4(x4)
        ], dim=1)
        out = self.fc(out)
        if x.shape[1] == out.shape[1]:
            out = out + x
        return out