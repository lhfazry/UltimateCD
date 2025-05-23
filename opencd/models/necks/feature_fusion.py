import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import NECKS
from mmcv.runner import BaseModule, auto_fp16


@NECKS.register_module()
class FeatureFusionNeck(BaseModule):
    """Feature Fusion Neck.

    Args:
        policy (str): The operation to fuse features. candidates 
            are `concat`, `sum`, `diff` and `Lp_distance`.
        in_channels (Sequence(int)): Input channels.
        channels (int): Channels after modules, before conv_seg.
        out_indices (tuple[int]): Output from which layer.
    """

    def __init__(self,
                 policy,
                 in_channels=None,
                 channels=None,
                 out_indices=(0, 1, 2, 3),
                 output_projection=False):
        super(FeatureFusionNeck, self).__init__()
        self.policy = policy
        self.in_channels = in_channels
        self.channels = channels
        self.out_indices = out_indices
        self.fp16_enabled = False

        if output_projection:
            self.output_projection = nn.Linear(in_channels, in_channels)
        else:
            self.output_projection = None

    @staticmethod
    def fusion(x1, x2, policy):
        """Specify the form of feature fusion"""
        
        _fusion_policies = ['concat', 'sum', 'diff', 'Lp_distance', 'L2_distance']
        assert policy in _fusion_policies, 'The fusion policies {} are ' \
            'supported'.format(_fusion_policies)
        
        if policy == 'concat':
            x = torch.cat([x1, x2], dim=1)
        elif policy == 'sum':
            x = x1 + x2
        elif policy == 'diff':
            x = x2 - x1
        elif policy == 'Lp_distance':
            x = torch.abs(x1 - x2)
        elif policy == 'L2_distance':
            x = torch.norm(x1 - x2, p=2, dim=(0,1,2,3))

        return x

    @auto_fp16()
    def forward(self, x1, x2):
        """Forward function."""

        assert len(x1) == len(x2), "The features x1 and x2 from the" \
            "backbone should be of equal length"
        outs = []
        for i in range(len(x1)):
            out = self.fusion(x1[i], x2[i], self.policy)

            if self.output_projection is not None:
                out = self.output_projection(out)

            outs.append(out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)