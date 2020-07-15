_base_ = ['_base_/pascal_voc_0712.py', '_base_/efficientdet.py']

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    gamma=0.5,
    step=[16, 24])

total_epochs = 30
