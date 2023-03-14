_base_ = 'efficientnet_b0.py'

model = dict(
    pretrained='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    backbone=dict(
        type='EfficientNet',
        model_name='efficientnet-b4'),
)