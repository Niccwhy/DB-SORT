#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None,
                 use_visible_head=False,        # [DB-SORT-VIS]
                 vis_loss_weight=0.7,           # [DB-SORT-VIS]
                 vis_assign_weight=0.3,         # [DB-SORT-VIS]
                 vis_warmup_steps=5000):        # [DB-SORT-VIS]
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80, use_visible_head=use_visible_head,
                             vis_loss_weight=vis_loss_weight,
                             vis_assign_weight=vis_assign_weight,
                             vis_warmup_steps=vis_warmup_steps)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg, loss_vis_iou = self.head(  # [DB-SORT-VIS]
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "vis_iou_loss": loss_vis_iou,  # [DB-SORT-VIS]
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
