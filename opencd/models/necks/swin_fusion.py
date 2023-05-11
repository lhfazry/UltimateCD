import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import NECKS
from mmcv.runner import BaseModule, auto_fp16
from models.backbones.swin_transformer import BasicLayer


@NECKS.register_module()
class SwinFusionNeck(BaseModule):
    def __init__(self,
                 in_channel=None,
                 out_indices=(0, 1, 2, 3)):
        super(SwinFusionNeck, self).__init__()
        self.in_channel = in_channel
        self.fp16_enabled = False
        self.out_indices = out_indices
        self.projection = nn.Linear(in_channel, in_channel)

    @staticmethod
    def fusion(x1, x2):
        x = torch.cat([x1, x2], dim=1)

        return x

    @auto_fp16()
    def forward(self, x1, x2):
        """Forward function."""

        assert len(x1) == len(x2), "The features x1 and x2 from the" \
            "backbone should be of equal shape"
    
        outs = []
        for i in range(len(x1)):
            if i == len(x1) - 1:
                out = self.fusion(x1[i], x2[i])
                out = self.projection(out)

            outs.append(out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)