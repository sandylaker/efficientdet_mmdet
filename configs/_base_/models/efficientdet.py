pretrained = 'pretrained/tf_efficientnet_b1_ns-99dd0c41.pth'   # Modify


model = dict(
    type='EfficientDet',
    pretrained=pretrained,
    scale=1,    # Modify
    backbone=dict(
        type='EfficientNet',
        in_channels=3,
        n_classes=20,
        se_rate=0.25,
        drop_connect_rate=0.3,
        frozen_stages=-1),  # Modify
    neck=dict(
        type='BiFPN',
        norm_cfg=dict(type='BN', momentum=0.01, eps=1e-3),
        upsample_cfg=dict(mode='nearest')),
    bbox_head=dict(
        type='EfficientHead',
        num_classes=20,
        num_ins=5,
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
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)