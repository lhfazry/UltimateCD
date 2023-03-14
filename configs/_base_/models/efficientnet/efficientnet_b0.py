# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='SiamEncoderDecoder',
    pretrained='https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    backbone=dict(
        type='EfficientNet',
        model_name='efficientnet-b0'),
    neck=dict(type='MixAttentionNeck'),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='TinyHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))