_base_ = 'focalnet_tiny_lrf.py'

model = dict(
    pretrained='./pretrained/focalnet_tiny_srf.pth',
    backbone=dict(
        type='FocalNet',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        drop_path_rate=0.2,
        focal_levels=[2, 2, 2, 2],
        focal_windows=[3, 3, 3, 3]),
)