_base_ = ['_base_/pascal_voc_0712.py', '_base_/efficientdet.py']

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3],   # 3 * 3= 9
)

total_epochs = 4

checkpoint_config = dict(interval=1, max_keep_ckpts=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]