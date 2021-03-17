_base_ = ['_base_/datasets/coco_detection.py', '_base_/models/efficientdet.py']

# optimizer
optimizer = dict(type='SGD', lr=0.96, momentum=0.9, weight_decay=4e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    min_lr=1e-5)
# actual epochs are multiplied by 3
runner = dict(type='EpochBasedRunner', max_epochs=300)

checkpoint_config = dict(interval=1, max_keep_ckpts=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, interval=1, warm_up=1000),
]