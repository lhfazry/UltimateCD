# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
#data_preprocessor = dict(
#    type='DualInputSegDataPreProcessor',
#    mean=[123.675, 116.28, 103.53] * 2,
#    std=[58.395, 57.12, 57.375] * 2,
#    bgr_to_rgb=True,
#    size_divisor=32,
#    pad_val=0,
#    seg_pad_val=255,
#    test_cfg=dict(size_divisor=32))
model = dict(
    type='SiamEncoderDecoder',
    #data_preprocessor=data_preprocessor,
    pretrained='pretrained/segformer.b0.512x512.ade.160k.pth',
    backbone=dict(
        type='lovit_b2',
        in_channels=3),
    neck=dict(type='FeatureFusionNeck', policy='diff'),
    decode_head=dict(
        type='SegformerConvHead',
        in_channels=[v for v in [64, 128, 320, 512]],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))