from models.builder import BACKBONES, NECKS, build_detector
from mmdet.models import HEADS
import torch
import os.path as osp
from mmcv.utils import Config


def test_backbone_neck_head():
    from mmcv.utils import build_from_cfg
    import torch

    # EfficientNet-B2, width_coef=1.1, depth_coef=1.2
    # outputs channels [48, 88, 120, 208, 352],
    cfg_backbone = dict(type='EfficientNet',
                        in_channels=3,
                        n_classes=20,
                        width_coefficient=1.1,
                        depth_coefficient=1.2,
                        se_rate=0.25,
                        dropout_rate=0.3,
                        frozen_stages=7)
    backbone = build_from_cfg(cfg_backbone, BACKBONES)

    # cfg_neck = dict(
    #     type='BiFPN',
    #     in_channels=[40, 80, 112, 192, 320],
    #     out_channels=112,
    #     num_outs=5,
    #     start_level=0,
    #     end_level=-1,
    #     stack=5)
    #
    # neck = build_from_cfg(cfg_neck, NECKS)

    images = torch.randn(1, 3, 768, 768)
    feats = backbone(images)  # noqa
    # feats = neck(feats)  # noqa
    for f in feats:
        print(f.shape)

    from mmdet.models.dense_heads import RetinaHead
    # cfg_head = dict(
    #     type='RetinaHead',
    #     num_classes=20,
    #     in_channels=112,
    #     stacked_convs=3,
    #     feat_channels=112,
    #     anchor_generator=dict(
    #         type='AnchorGenerator',
    #         octave_base_scale=4,
    #         scales_per_octave=3,
    #         ratios=[0.5, 1.0, 2.0],
    #         strides=[8, 16, 32, 64, 128]),
    #     bbox_coder=dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[.0, .0, .0, .0],
    #         target_stds=[1.0, 1.0, 1.0, 1.0]),
    #     loss_cls=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=1.0),
    #     loss_bbox=dict(type='L1Loss', loss_weight=1.0))
    #
    # head = build_from_cfg(cfg_head, HEADS)
    #
    # cls_scores, bbox_preds = head(feats)  # noqa


def test_build_detector():
    ROOT = osp.dirname(osp.dirname(__file__))
    cfg = Config.fromfile(osp.join(ROOT, 'configs/efficientdet.py'))

    device = torch.device('cuda')
    model = build_detector(cfg.model, cfg.train_cfg, cfg.test_cfg).to(device)

    img = torch.randn(1, 3, 224, 224).to(device)
    gt_bboxes = [torch.tensor([[20.0, 20.0, 50.0, 50.0],
                               [70.0, 70.0, 150.0, 150.0]]).to(device)]
    gt_labels = [torch.tensor([1, 2]).to(device)]
    img_metas = [dict(img_shape=(224, 224),
                      ori_shape=(224, 224),
                      pad_shape=(224, 224),
                      img_norm_cfg=dict(
                          mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                      scale_factor=1.0)]
    losses = model(img, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
    print(type(losses))
    for key, loss in losses.items():
        if isinstance(loss, torch.Tensor):
            print(key, loss.shape)
        else:
            print(key, loss)


if __name__ == '__main__':
    test_backbone_neck_head()
    # test_build_detector()