# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='SiamEncoderDecoder',
    pretrained='./pretrained/swin_base_patch4_window7_224.pth',
    backbone=dict(
        type='SwinTransformer',
        img_size=256, 
        patch_size=4, 
        in_chans=3,
        embed_dim=128, 
        depths=[2, 2, 18, 2], 
        num_heads=[4, 8, 16, 32],
        window_size=7),
    decode_head=dict(
        type='SwinHead',
        img_size=256, 
        patch_size=4, 
        in_chans=3, 
        embed_dim=128, 
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))