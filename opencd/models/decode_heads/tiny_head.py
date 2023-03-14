# Copyright (c) Open-CD. All rights reserved.
from dis import dis
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule, Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from typing import List, Optional
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmcv.runner import BaseModule, auto_fp16, Sequential
from torch import Tensor, reshape, stack
from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    PReLU,
    ModuleList,
    Sequential,
    Upsample,
    Sigmoid
)

class PixelwiseLinear(Module):
    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)

class UpMask(Module):
    def __init__(
        self,
        scale_factor: float,
        nin: int,
        nout: int,
    ):
        super().__init__()
        self._upsample = Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self._convolution = Sequential(
            Conv2d(nin, nin, 3, 1, groups=nin, padding=1),
            PReLU(),
            InstanceNorm2d(nin),
            Conv2d(nin, nout, kernel_size=1, stride=1),
            PReLU(),
            InstanceNorm2d(nout),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)
    
@HEADS.register_module()
class TinyHead(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self._upmask = ModuleList(
            [
                UpMask(2, 56, 64),
                UpMask(2, 64, 64),
                UpMask(2, 64, 32),
            ]
        )

        # Final classification layer:
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], Sigmoid())
        
    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        upping = inputs[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._upmask[i](upping, inputs[j])

        return self._classify(upping)
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
    
        return losses
    
    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        seg_logits = self.forward(inputs)
        return seg_logits
