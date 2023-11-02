from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F
import timm


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, num_features: int = 256):
        super().__init__()

        self.levels = (3, 4, 5, 6, 7)

        # Reduction
        self.p6 = nn.Conv2d(512, num_features, kernel_size=3, stride=2, padding=1)
        self.p7_relu = nn.ReLU(inplace=True)
        self.p7 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1)

        # Expansion
        self.p5_1 = nn.Conv2d(512, num_features, kernel_size=1)
        self.p5_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.p4_1 = nn.Conv2d(256, num_features, kernel_size=1)
        self.p4_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(128, num_features, kernel_size=1)
        self.p3_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def __call__(self, c3, c4, c5):
        # Reduction
        p6 = self.p6(c5)

        p7 = self.p7_relu(p6)
        p7 = self.p7(p7)

        # Expansion
        p5 = self.p5_1(c5)
        p5_up = F.interpolate(p5, scale_factor=2)
        p5 = self.p5_2(p5)

        p4 = self.p4_1(c4) + p5_up
        p4_up = F.interpolate(p4, scale_factor=2)
        p4 = self.p4_2(p4)

        p3 = self.p3_1(c3) + p4_up
        p3 = self.p3_2(p3)

        return p3, p4, p5, p6, p7


class DetectionHead(nn.Module):
    def __init__(
        self,
        num_channels_per_anchor: int,
        num_anchors: int = 9,
        num_features: int = 256,
    ):
        super().__init__()

        self.num_anchors = num_anchors

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )

        self.out_conv = nn.Conv2d(
            num_features,
            num_anchors * num_channels_per_anchor,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        for i in range(4):
            x = self.conv_blocks[i](x)
        x = self.out_conv(x)

        bs, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = x.reshape(bs, w * h * self.num_anchors, -1)
        return x


class AnchorGenerator:
    def __init__(self, levels):
        ratios = torch.tensor([0.5, 1.0, 2.0])
        scales = torch.tensor([2**0, 2 ** (1 / 3), 2 ** (2 / 3)])
        self.num_anchors = ratios.shape[0] * scales.shape[0]
        self.strides = [2**level for level in levels]

        self.anchors = []
        for level in levels:
            base_length = 2 ** (level + 2)

            scaled_length = base_length * scales
            anchor_areas = scaled_length**2

            anchor_widths = (anchor_areas / ratios.unsqueeze(1)) ** 0.5
            anchor_heights = anchor_widths * ratios.unsqueeze(1)

            anchor_widths = anchor_widths.flatten()
            anchor_heights = anchor_heights.flatten()

            anchor_xmin = -0.5 * anchor_widths
            anchor_ymin = -0.5 * anchor_heights
            anchor_xmax = 0.5 * anchor_widths
            anchor_ymax = 0.5 * anchor_heights

            level_anchors = torch.stack((anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax), dim=1)

            self.anchors.append(level_anchors)

    @torch.no_grad()
    def generate(self, feature_sizes: Sequence[torch.Size]):
        anchors = []
        for stride, level_anchors, feature_size in zip(self.strides, self.anchors, feature_sizes):
            height, width = feature_size

            xs = (torch.arange(width) + 0.5) * stride
            ys = (torch.arange(height) + 0.5) * stride

            grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()

            anchor_xmin = (grid_x.unsqueeze(1) + level_anchors[:, 0]).flatten()
            anchor_ymin = (grid_y.unsqueeze(1) + level_anchors[:, 1]).flatten()
            anchor_xmax = (grid_x.unsqueeze(1) + level_anchors[:, 2]).flatten()
            anchor_ymax = (grid_y.unsqueeze(1) + level_anchors[:, 3]).flatten()

            level_anchors = torch.stack([anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax], dim=1)
            anchors.append(level_anchors)

        anchors = torch.cat(anchors)
        return anchors


class RetinaNet(nn.Module):
    def __init__(self, num_classes: int, backbone="resnet18"):
        super().__init__()

        self.backbone: nn.Module
        self.fpn = FeaturePyramidNetwork()
        self.anchor_generator = AnchorGenerator(self.fpn.levels)

        self.cls_head = DetectionHead(num_classes, self.anchor_generator.num_anchors)
        self.box_head = DetectionHead(4, self.anchor_generator.num_anchors)

        self._reset_parameters()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)

    def _reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")

        prior = torch.tensor(0.01)
        nn.init.zeros_(self.cls_head.out_conv.weight)
        nn.init.constant_(self.cls_head.out_conv.bias, -((1.0 - prior) / prior).log())

        nn.init.zeros_(self.box_head.out_conv.weight)
        nn.init.zeros_(self.box_head.out_conv.bias)

    def forward(self, x):
        cs = self.backbone(x)[-3:]
        ps = self.fpn(*cs)

        preds_class = torch.cat(list(map(self.cls_head, ps)), dim=1)
        preds_box = torch.cat(list(map(self.box_head, ps)), dim=1)

        feature_sizes = [p.shape[2:] for p in ps]
        anchors = self.anchor_generator.generate(feature_sizes)
        anchors = anchors.to(x.device)

        return preds_class, preds_box, anchors
