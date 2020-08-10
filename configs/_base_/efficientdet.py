pretrained = '/home/lfc199471/PycharmProjects/efficientdet_mmdet/pretrained/' \
             'efficientnet-b1-f1951068.pth'   # Modify


model = dict(
    type='EfficientDet',
    pretrained=pretrained,
    backbone=dict(
        type='EfficientNet',
        in_channels=3,
        n_classes=20,
        width_coefficient=1.0,  # Modify
        depth_coefficient=1.1,  # Modify
        se_rate=0.25,
        dropout_rate=0.2,   # Modify
        drop_connect_rate=0.3,
        frozen_stages=1),  # Modify
    neck=dict(
        type='BiFPN',
        in_channels=[40, 112, 320],  # Modify
        out_channels=88,   # Modify
        num_outs=5,
        start_level=0,
        end_level=-1,
        norm_cfg=dict(type='BN', momentum=0.01, eps=1e-3),
        stack=4),   # Modify
    bbox_head=dict(
        type='RetinaHead',
        num_classes=20,
        in_channels=88,    # Modify
        stacked_convs=3,    # Modify
        feat_channels=88,  # Modify
        norm_cfg=dict(type='BN', momentum=0.01, eps=1e-3),
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)))

# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)
total_epochs = 12

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
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
