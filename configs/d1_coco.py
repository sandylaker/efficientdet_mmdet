_base_ = ['_base_/datasets/coco_detection.py', '_base_/models/efficientdet.py']

model = dict(
    scale=1,    # Modify
    backbone=dict(
        n_classes=1000),
    bbox_head=dict(
        num_classes=80))

# optimizer
optimizer = dict(type='SGD', lr=0.18, momentum=0.9, weight_decay=4e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.001,
    warmup_by_epoch=True,
    min_lr=1e-5)

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
    dict(type='EMAHook', momentum=0.0002, interval=1, warm_up=4000),
]
