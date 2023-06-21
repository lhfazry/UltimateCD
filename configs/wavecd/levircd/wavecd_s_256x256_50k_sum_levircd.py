_base_ = [
    '../../_base_/models/siam_upernet_wavevit.py', '../../_base_/datasets/levir_cd256.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_50k.py'
]

embed_dims=[64, 128, 320, 448]

model = dict(
    pretrained='./pretrained/wavevit_s.pth',
    backbone=dict(
        stem_hidden_dim=32, 
        embed_dims=embed_dims,
        num_heads=[2, 4, 10, 14], 
        drop_path_rate=0.3, #0.2, 
        depths=[3, 4, 6, 3]
    ),
    neck=dict(type='FeatureFusionNeck', policy='sum'),
    decode_head=dict(
        in_channels=[v for v in embed_dims],
        num_classes=2
    ),
    auxiliary_head=dict(
        in_channels=embed_dims[2],
        num_classes=2
    ))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#crop_size = (256, 256)

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
work_dir = './work_dirs/wavecd/levircd/wavecd_s_256x256_50k_sum_levircd'