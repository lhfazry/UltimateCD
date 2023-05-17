_base_ = [
    '../../_base_/datasets/whu256.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_100k.py'
]

in_channels = [96, 192, 384, 768]
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='SiamEncoderDecoder',
    pretrained='./pretrained/focalnet_tiny_lrf.pth',
    backbone=dict(
        type='FocalNet',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        drop_path_rate=0.0,
        focal_levels=[3, 3, 3, 3],
        focal_windows=[3, 3, 3, 3]),
    neck=dict(type='FeatureFusionNeck', policy='Lp_distance'),
    decode_head=dict(
        type='UPerHead',
        in_channels=[v for v in in_channels],
        in_index=[0, 1, 2, 3],
        channels=128,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
#data=dict(samples_per_gpu=2)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
fp16 = dict()
work_dir = './work_dirs/focalcd/whu/focalcd_t_256x256_50k_absdiff_noaux_nodrop_whu'