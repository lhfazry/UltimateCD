# dataset settings
dataset_type = 'CDD_Dataset'
data_root = 'datasets/cdd'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#crop_size = (128, 128)#(256, 256)

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    #dict(type='MultiImgRandomCrop', crop_size=crop_size),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(type='MultiImgNormalize', **img_norm_cfg),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),  
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(
        type='MultiImgMultiScaleFlipAug',
        img_scale=(256,256),#(1024, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='MultiImgResize', keep_ratio=True),
            dict(type='MultiImgRandomFlip'),
            dict(type='MultiImgNormalize', **img_norm_cfg),
            dict(type='MultiImgImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir='train/OUT',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir='val/OUT',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='test/OUT',
        pipeline=test_pipeline))