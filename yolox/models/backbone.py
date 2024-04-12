#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import (
    BaseConv,
    CSPLayer,
    DWConv,
    Focus,
    ResLayer,
    SPPBottleneck,
    RepNCSPELAN4,
    ADown
)


class GelanCBackbone(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("p3", "p4", "p5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        self.p3 = nn.Sequential(
            BaseConv(3, 64, 3, 2),
            BaseConv(64, 128, 3, 2),
            RepNCSPELAN4(128, 256, 64, 1),
            ADown(256, 256),
            RepNCSPELAN4(256, 512, 128, 1),
        )
        self.p4 = nn.Sequential(
            ADown(512, 512),
            RepNCSPELAN4(512, 512, 256, 1),
        )
        self.p5 = nn.Sequential(
            ADown(512, 512),
        )

    def forward(self, x):
        outputs = {}
        x = self.p3(x)
        outputs["p3"] = x
        x = self.p4(x)
        outputs["p4"] = x
        x = self.p5(x)
        outputs["p5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
