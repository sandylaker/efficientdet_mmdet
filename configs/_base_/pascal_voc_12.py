dataset_type = 'MyVOCDataset'
# Remember to change this root
data_root = '/home/lfc199471/data/VOCdevkit/'
img_size = 512  # Modify
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(img_size, img_size)], keep_ratio=True),   # Modify
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(img_size, img_size)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),   # Modify
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(img_size, img_size)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

batch_size = 24     # Modify

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Main/train.txt',
        img_prefix=data_root + 'VOC2012/',
        pipeline=train_pipeline,),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2012/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2012/',
        pipeline=test_pipeline,),
)

evaluation=dict(interval=2, metric='mAP')
