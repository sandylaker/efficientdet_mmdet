# optimizer
optimizer = dict(type='SGD', lr=0.06, momentum=0.9, weight_decay=4e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=0.001,
    min_lr=1e-5)
total_epochs = 300