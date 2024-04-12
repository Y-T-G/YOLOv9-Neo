#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .backbone import GelanCBackbone
from .network_blocks import (
    BaseConv,
    SPPElanBottleneck,
    DWConv,
    RepNCSPELAN4,
    ADown
)


class YOLOPAFPN(nn.Module):
    """
    YOLOv9 FPN.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("p3", "p4", "p5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = GelanCBackbone(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.CSPElan0 = RepNCSPELAN4(512, 512, 256, 1)
        self.elan_spp = SPPElanBottleneck(512, 512)
        self.CSPElan1 = RepNCSPELAN4(1024, 512, 256, 1)
        self.CSPElan2 = RepNCSPELAN4(1024, 256, 128, 1)
        self.down_pn0 = ADown(256,256)
        self.CSPElan3 = RepNCSPELAN4(768, 512, 256, 1)
        self.down_pn1 = ADown(512,512)
        self.CSPElan4 = RepNCSPELAN4(1024, 512, 256, 1)

    def forward(self, x):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(x)
        features = [out_features[f] for f in self.in_features]
        [p3, p4, p5] = features

        f_out = self.CSPElan0(p5)
        fpn_out0 = self.elan_spp(f_out)

        f_out = self.upsample(fpn_out0)
        f_out = torch.cat([f_out, p4], 1)
        fpn_out1 = self.CSPElan1(f_out)

        f_out = self.upsample(fpn_out1)
        f_out = torch.cat([f_out, p3], 1)
        pan_out0 = self.CSPElan2(f_out)

        p_out = self.down_pn0(pan_out0)
        p_out = torch.cat([p_out, fpn_out1], 1)
        pan_out1 = self.CSPElan3(p_out)

        p_out = self.down_pn1(pan_out1)
        p_out = torch.cat([p_out, fpn_out0], 1)
        pan_out2 = self.CSPElan4(p_out)

        outputs = (pan_out0, pan_out1, pan_out2)
        return outputs
