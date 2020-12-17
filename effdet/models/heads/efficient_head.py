import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init
from ..necks import DepthwiseSeparableConvModule
from mmdet.models import AnchorHead


class EfficientHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 stacked_convs=4,
                 norm_cfg=dict(type='BN', momentum=0.01, eps=1e-3),
                 act_cfg=dict(type='Swish'),
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        super(EfficientHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            anchor_generator=anchor_generator,
            **kwargs)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            # separate normalization layers are applied to cls and reg branches
            self.cls_convs.append(
                nn.Sequential(
                    DepthwiseSeparableConvModule(
                        chn,
                        self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dw_norm_cfg=None,
                        dw_act_cfg=None,
                        pw_norm_cfg=self.norm_cfg,
                        pw_act_cfg=self.act_cfg)))
            self.reg_convs.append(
                nn.Sequential(
                    DepthwiseSeparableConvModule(
                        chn,
                        self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dw_norm_cfg=None,
                        dw_act_cfg=None,
                        pw_norm_cfg=self.norm_cfg,
                        pw_act_cfg=self.act_cfg)))
        # no activation and normalization applied to the last layer
        # apply bias to the last layer
        self.efficient_cls = DepthwiseSeparableConvModule(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1,
            dw_norm_cfg=None,
            dw_act_cfg=None,
            pw_norm_cfg=None,
            pw_act_cfg=None,
            bias=True)
        self.efficient_reg = DepthwiseSeparableConvModule(
            self.feat_channels,
            self.num_anchors * 4,
            3,
            padding=1,
            dw_norm_cfg=None,
            dw_act_cfg=None,
            pw_norm_cfg=None,
            pw_act_cfg=None,
            bias=True)

    def init_weights(self):
        for m in self.cls_convs:
            ds_conv = m[0]
            normal_init(ds_conv.depthwise_conv.conv, std=0.01)
            normal_init(ds_conv.pointwise_conv.conv, std=0.01)
        for m in self.reg_convs:
            ds_conv = m[0]
            normal_init(ds_conv.depthwise_conv.conv, std=0.01)
            normal_init(ds_conv.pointwise_conv.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.efficient_cls.depthwise_conv.conv, std=0.01)
        normal_init(self.efficient_cls.pointwise_conv.conv, std=0.01, bias=bias_cls)

        normal_init(self.efficient_reg.depthwise_conv.conv, std=0.01)
        normal_init(self.efficient_reg.pointwise_conv.conv, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.efficient_cls(cls_feat)
        bbox_pred = self.efficient_reg(reg_feat)
        return cls_score, bbox_pred
