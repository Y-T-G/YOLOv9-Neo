#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if not act:
            self.act = nn.Identity()
        else:
            self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        groups=1,
        kernel=(3, 3),
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(
            in_channels, hidden_channels, kernel[0], stride=1, act=act
        )
        self.conv2 = Conv(hidden_channels, out_channels, kernel[1], stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class RepBottleneck(Bottleneck):
    # Repeating standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        groups=1,
        kernel=(3, 3),
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__(
            in_channels,
            out_channels,
            shortcut,
            groups,
            kernel,
            expansion,
            depthwise,
            act,
        )
        hidden_channels = int(out_channels * expansion)
        self.conv1 = RepConv(in_channels, hidden_channels, kernel[0], stride=1, act=act)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class RepConv(nn.Module):
    # https://github.com/DingXiaoH/RepVGG
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        act=nn.ReLU(),
    ):
        super().__init__()
        self.conv1 = BaseConv(
            in_channels, out_channels, kernel_size, stride, groups=groups, act=False
        )
        self.conv2 = BaseConv(
            in_channels, out_channels, 1, stride, groups=groups, act=False
        )
        self.act = get_activation(act) if isinstance(act, str) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    # to be implement
    # def fuse_convs(self):


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, 1))


class SPPElanBottleneck(SPPBottleneck):
    """Spatial pyramid pooling ELAN layer."""

    def __init__(self, in_channels, out_channels, ks=5, activation="silu"):
        super().__init__(in_channels, out_channels, activation=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for _ in range(3)]
        )

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, 1))


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut,
                expansion=1.0,
                depthwise=depthwise,
                act=act,
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class RepCSPLayer(CSPLayer):
    """CSPLayer wit RepBottleneck"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(RepCSPLayer, self).__init__(
            in_channels, out_channels, n, shortcut, expansion, depthwise, act
        )
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            RepBottleneck(
                hidden_channels,
                hidden_channels,
                shortcut,
                expansion=1.0,
                depthwise=depthwise,
                act=act,
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(
        self,
        in_channels,
        out_channels,
        repeat=1,
        shortcut=True,
        groups=1,
        expansion=0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv3 = BaseConv(
            int(2 * hidden_channels), out_channels, 1, 1
        )  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(
                RepBottleneck(
                    hidden_channels, hidden_channels, shortcut, groups, expansion=1.0
                )
                for _ in range(repeat)
            )
        )

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN"""

    def __init__(
        self, in_channels, out_channels, med_channels, groups=1, shortcut=True
    ):
        super().__init__()
        conv1_out = med_channels * 2
        self.bl_out = med_channels
        self.conv1 = BaseConv(in_channels, conv1_out, 1, 1)
        self.conv2 = nn.Sequential(
            RepCSPLayer(med_channels, med_channels, shortcut),
            BaseConv(med_channels, med_channels, 3, 1),
        )
        self.conv3 = nn.Sequential(
            RepCSPLayer(med_channels, med_channels, shortcut),
            BaseConv(med_channels, med_channels, 3, 1),
        )
        self.conv4 = BaseConv(conv1_out + (2 * med_channels), out_channels, 1, 1)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.conv2, self.conv3])
        return self.conv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.conv1(x).split((self.bl_in, self.bl_in), 1))
        y.extend(m(y[-1]) for m in [self.conv2, self.conv3])
        return self.conv4(torch.cat(y, 1))


class CSPELAN(nn.Module):
    # ELAN
    def __init__(
        self,
        in_channels,
        out_channels,
        med_channels,
        elan_repeat=2,
        cb_repeat=2,
        expansion=0.5,
    ):

        super().__init__()

        h_channels = med_channels // 2
        self.conv1 = BaseConv(in_channels, med_channels, 1, 1)
        self.convb = nn.ModuleList(
            RepCSPLayer(h_channels, h_channels, n=cb_repeat, expansion=expansion)
            for _ in range(elan_repeat)
        )
        self.conv2 = BaseConv(
            med_channels + (elan_repeat * h_channels), out_channels, 1, 1
        )

    def forward(self, x):

        y = list(self.conv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in self.convb)

        return self.conv2(torch.cat(y, 1))


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), RepConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
