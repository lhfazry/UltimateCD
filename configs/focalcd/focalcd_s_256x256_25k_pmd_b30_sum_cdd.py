_base_ = [
    '../_base_/models/siam_upernet_focalnet.py', '../_base_/datasets/cdd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_25k.py'
]

in_channels = [96, 192, 384, 768]

model = dict(
    backbone=dict(
        init_cfg = dict(type='Pretrained', checkpoint='./pretrained/focalnet_small_lrf.pth'),
        embed_dim=96,
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,    
        focal_windows=[9, 9, 9, 9],
        focal_levels=[3, 3, 3, 3],
    ),
    neck=dict(type='FeatureFusionNeck', policy='sum'),
    decode_head=dict(
        in_channels=[v for v in in_channels],
        num_classes=2
    ),
    auxiliary_head=dict(
        in_channels=in_channels[2],
        num_classes=2
    ))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgExchangeTime', prob=0.5),

    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=30),
    
    dict(type='MultiImgNormalize', **img_norm_cfg),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline)
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
work_dir = './work_dirs/focalcd/focalcd_s_256x256_25k_pmd_b30_sum_cdd'