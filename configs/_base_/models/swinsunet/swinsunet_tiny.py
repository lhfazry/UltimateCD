# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
embed_dim = 96
depths = [2, 2, 6, 2]
num_heads = [3, 6, 12, 24]

model = dict(
    type='SiamEncoderDecoder',
    #pretrained='./pretrained/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        type='SwinTransformer',
        in_channels=3,
        embed_dims=embed_dim,
        num_heads=num_heads,
        window_size=7,
        patch_size=4,  
        depths=depths),
    neck=dict(
        type='SwinFusionNeck', 
        in_channels=2 * embed_dim * 2 ** (len(depths) - 1),
        out_channels=embed_dim * 2 ** (len(depths) - 1)
    ),
    decode_head=dict(
        type='SwinHead',
        in_channels=embed_dim * 2 ** (len(depths) - 1),
        num_classes=2,
        patch_size=4, 
        embed_dims=embed_dim, 
        depths=depths, 
        num_heads=num_heads,
        window_size=7,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))