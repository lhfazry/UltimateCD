_base_ = [
    '../_base_/models/bit_r18.py', '../_base_/datasets/levir_cd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_50k.py'
]
model = dict(
    neck=dict(type='FeatureFusionNeck', policy='concat'),
    decode_head=dict(
        input_transform='resize_concat',
        in_index=[0,1,2,3],
        in_channels=[64,128,256,512],
        num_classes=2,
        pre_upsample=1),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=2))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgResize', img_scale=(512, 512)),
    dict(type='MultiImgRandomCrop', crop_size=crop_size),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgExchangeTime', prob=0.5),
    
    dict(type='MultiImgNormalize', **img_norm_cfg),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(
        type='MultiImgMultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='MultiImgResize', keep_ratio=True),
            dict(type='MultiImgRandomFlip'),
            dict(type='MultiImgNormalize', **img_norm_cfg),
            dict(type='MultiImgImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.99, weight_decay=0.0005)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-8, by_epoch=True)
# runtime settings
#runner = dict(type='EpochBasedRunner', max_epochs=200)
#checkpoint_config = dict(by_epoch=True, interval=10)
#evaluation = dict(interval=50, metric=['mFscore', 'mIoU'], pre_eval=True, save_best='Fscore.changed', greater_keys=['Fscore'])
work_dir = './work_dirs/bit/bit_r18_256x256_50k_levircd'