_base_ = [
    '../../_base_/datasets/cdd.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_10k.py'
]

embed_dims=[64, 128, 320, 448]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='SiamEncoderDecoder',
    pretrained=None,
    backbone=dict(
        pretrained='./pretrained/wavevit_s.pth',
        type='WaveViT',
        stem_hidden_dim=32, 
        embed_dims=embed_dims,
        num_heads=[2, 4, 10, 14], 
        drop_path_rate=0.3, #0.2, 
        depths=[3, 4, 6, 3],
        locality_ffn=True),
    neck=dict(type='FeatureFusionNeck', policy='sum'),
    decode_head=dict(
        type='SegFormerMLPHead',
        in_channels=[v for v in embed_dims],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=embed_dims[2],
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

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (128, 128)#(256, 256)

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgExchangeTime', prob=0.5),
    
    dict(type='MultiImgNormalize', **img_norm_cfg),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline)
)

workflow = [('train', 1), ('val', 1)]

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
work_dir = './work_dirs/wavecd/ablation/wavecd_s_128x128_10k_sum_locality_mlp_cdd'