from backbones.wavevit import WaveViT
from functools import partial
import torch
from torch import nn

wavevit = WaveViT(
    in_chans=3, 
    stem_hidden_dim = 64,
    embed_dims=[64, 128, 320, 512],
    num_heads=[2, 4, 10, 16], 
    mlp_ratios=[8, 8, 4, 4], 
    depths=[3, 4, 12, 3],
    sr_ratios = [4, 2, 1, 1], 
    norm_layer = partial(nn.LayerNorm, eps=1e-6), 
    token_label=True
)

input = torch.rand((8, 3, 244, 244), dtype=torch.float32)
#model = wavevit.cuda()

output = wavevit(input)
print(output.shape)