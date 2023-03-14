import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional
from mmseg.models.builder import NECKS
from mmcv.runner import BaseModule, auto_fp16
from torch import Tensor, reshape, stack

from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    PReLU,
    ModuleList,
    Sequential
)
    
class MixingBlock(Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            PReLU(),
            InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))

        # Mixing:
        return self._convmix(mixed)

class MixingMaskAttentionBlock(Module):
    """use the grouped convolution to make a sort of attention"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        fin: List[int],
        fout: List[int],
        generate_masked: bool = False,
    ):
        super().__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)

        return (
            z
            if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )
    
@NECKS.register_module()
class MixAttentionNeck(BaseModule):
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
                 out_indices=(0, 1, 2, 3)):
        super(MixAttentionNeck, self).__init__()
        #self.policy = policy
        #self.in_channels = in_channels
        #self.channels = channels
        self.out_indices = out_indices
        #self.fp16_enabled = False

        self._mixing_mask = ModuleList(
            [
                MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1]),
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingBlock(112, 56)
            ]
        )

    @auto_fp16()
    def forward(self, x1, x2):
        """Forward function."""

        assert len(x1) == len(x2), "The features x1 and x2 from the" \
            "backbone should be of equal length"

        outs = []
        for i in range(len(x1)):
            out = self._mixing_mask[i](x1[i], x2[i])
            outs.append(out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)