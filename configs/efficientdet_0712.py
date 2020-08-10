_base_ = ['_base_/pascal_voc_0712.py', '_base_/efficientdet.py']

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=2e-4)

lr_config = dict(
    _delete_=True,
    policy='CosineAnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=5e-4
    )

total_epochs = 15
