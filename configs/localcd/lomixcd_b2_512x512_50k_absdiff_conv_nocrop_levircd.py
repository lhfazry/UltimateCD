_base_ = ['../_base_/models/localcd/lomixcd_b2.py', '../_base_/datasets/levir_cd512.py',
        '../_base_/default_runtime.py', '../_base_/schedules/schedule_50k.py']


model = dict(
    neck=dict(type='FeatureFusionNeck', policy='Lp_distance')
)

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
work_dir = './work_dirs/localcd/lomixcd_b2_512x512_50k_absdiff_conv_nocrop_levircd'