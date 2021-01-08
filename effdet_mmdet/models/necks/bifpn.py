import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from .utils import WeightedAdd, DepthwiseSeparableConvModule
from ..backbones.efficient_net_utils import Swish
from functools import partial


class BiFPN(nn.Module):
    def __init__(self,
                 in_channels_list: list,
                 out_channels:int,
                 stack: int,
                 norm_cfg: dict = dict(type='BN', momentum=0.01, eps=1e-3),
                 upsample_cfg: dict = dict(mode='nearest')):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels_list, list)
        assert len(in_channels_list) == 3, f"Length of input feature maps list should be 3, " \
                                           f"got {len(in_channels_list)}"

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.upsample_cfg = upsample_cfg
        self.stack = stack
        self.swish = Swish()

        self.bfpn_list = nn.ModuleList()
        for i in range(self.stack):
            self.bfpn_list.append(
                SingleBiFPN(
                    out_channels=self.out_channels,
                    stack_idx=i,
                    # in_channels_list is only useful for stack_idx = 0
                    in_channels_list=self.in_channels_list,
                    norm_cfg=norm_cfg,
                    upsample_cfg=upsample_cfg))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        for layer in self.bfpn_list:
            inputs = layer(inputs)
        return inputs


class SingleBiFPN(nn.Module):
    def __init__(self,
                 out_channels: int,
                 stack_idx: int,
                 in_channels_list: list = None,
                 norm_cfg: dict = None,
                 upsample_cfg: dict = dict(mode='nearest'),
                 eps:float = 1e-4):
        super(SingleBiFPN, self).__init__()

        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.eps = eps
        self.stack_idx = stack_idx
        self.upsample_cfg = upsample_cfg
        self.swish = Swish()
        self.upsample_func = partial(F.interpolate, scale_factor=2, **self.upsample_cfg)
        self.downsample_func = partial(F.max_pool2d, kernel_size=3, stride=2, padding=1)

        # build layers, only build conv modules for input when stack index is 0
        if self.stack_idx == 0:
            self.in_channels_list = in_channels_list
            self._build_in_conv_layers()
        self._build_td_bu_layers()

    def _build_in_conv_layers(self):
        channels_p3, channels_p4, channels_p5 = self.in_channels_list
        in_conv_channels_list = (channels_p3, channels_p4, channels_p4, channels_p5, channels_p5,
                                 channels_p5)
        in_conv_names = ('conv_p3_in', 'conv_p4_in_1', 'conv_p4_in_2', 'conv_p5_in_1',
                         'conv_p5_in_2', 'conv_p6_in')
        self.in_convs = nn.ModuleDict()
        for name, in_channels in zip(in_conv_names, in_conv_channels_list):
            # no built-in activation, the output will be manually activated by swish
            conv_module = ConvModule(
                in_channels,
                self.out_channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
            self.in_convs.update({name: conv_module})

    def _build_td_bu_layers(self):
        """Build top-down and bottom-up paths"""
        # Names of weighted addition modules in top down path (2 inputs)
        wadd_names_td = ('wadd_p6_td', 'wadd_p5_td', 'wadd_p4_td', 'wadd_p3_td')
        self.wadd_td = nn.ModuleDict()
        for name_td in wadd_names_td:
            wadd_module = WeightedAdd(num_inputs=2)
            self.wadd_td.update({name_td: wadd_module})

        # Names of weighted addition modules in bottom up path (3 inputs)
        wadd_names_bu = ('wadd_p4_bu', 'wadd_p5_bu', 'wadd_p6_bu')
        self.wadd_bu = nn.ModuleDict()
        for name_bu in wadd_names_bu:
            wadd_module = WeightedAdd(num_inputs=3)
            self.wadd_bu.update({name_bu: wadd_module})
        # Weighted addition of p7 in bottom up path takes 2 inputs
        self.wadd_bu.update({'wadd_p7_bu': WeightedAdd(num_inputs=2)})

        # Names of depthwise separable convs in top down path
        conv_names_td = ('conv_p6_td', 'conv_p5_td', 'conv_p4_td', 'conv_p3_td')
        self.conv_td = nn.ModuleDict()
        for name_td in conv_names_td:
            # no built-in activation, the output will be manually activated by swish
            conv_module = DepthwiseSeparableConvModule(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                dw_norm_cfg=None,
                dw_act_cfg=None,
                pw_norm_cfg=self.norm_cfg,
                pw_act_cfg=None)
            self.conv_td.update({name_td: conv_module})

        # Names of depthwise separable convs in bottom up path
        conv_names_bu = ('conv_p4_bu', 'conv_p5_bu', 'conv_p6_bu', 'conv_p7_bu')
        self.conv_bu = nn.ModuleDict()
        for name_bu in conv_names_bu:
            # no built-in activation, the output will be manually activated by swish
            conv_module = DepthwiseSeparableConvModule(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                dw_norm_cfg=None,
                dw_act_cfg=None,
                pw_norm_cfg=self.norm_cfg,
                pw_act_cfg=None)
            self.conv_bu.update({name_bu: conv_module})

    def forward(self, feats):
        if self.stack_idx == 0:
            c3, c4, c5 = feats
            p3_in = c3
            p4_in = c4
            p5_in = c5
            p6_in = self.in_convs['conv_p6_in'](p5_in)
            p6_in = self.downsample_func(p6_in)
            p7_in = self.downsample_func(p6_in)

            p4_in_1 = self.in_convs['conv_p4_in_1'](p4_in)
            p5_in_1 = self.in_convs['conv_p5_in_1'](p5_in)

            p4_in_2 = self.in_convs['conv_p4_in_2'](p4_in)
            p5_in_2 = self.in_convs['conv_p5_in_2'](p5_in)

            p3_in = self.in_convs['conv_p3_in'](p3_in)
        else:
            p3_in, p4_in_1, p5_in_1, p6_in, p7_in = feats
            p4_in_2 = p4_in_1.clone()
            p5_in_2 = p5_in_1.clone()

        # Top down path
        p7_u = self.upsample_func(p7_in, scale_factor=2, mode='nearest')
        p6_td = self.wadd_td['wadd_p6_td']([p6_in, p7_u])
        p6_td = self.swish(p6_td)
        p6_td = self.conv_td['conv_p6_td'](p6_td)

        p6_u = self.upsample_func(p6_td)
        p5_td = self.wadd_td['wadd_p5_td']([p5_in_1, p6_u])
        p5_td = self.swish(p5_td)
        p5_td = self.conv_td['conv_p5_td'](p5_td)

        p5_u = self.upsample_func(p5_td)
        p4_td = self.wadd_td['wadd_p4_td']([p4_in_1, p5_u])
        p4_td = self.swish(p4_td)
        p4_td = self.conv_td['conv_p4_td'](p4_td)

        p4_u = self.upsample_func(p4_td)
        p3_td = self.wadd_td['wadd_p3_td']([p3_in, p4_u])
        p3_td = self.swish(p3_td)
        p3_td = self.conv_td['conv_p3_td'](p3_td)

        # Bottom up path
        p3_d = self.downsample_func(p3_td)
        p4_bu = self.wadd_bu['wadd_p4_bu']([p4_in_2, p4_td, p3_d])
        p4_bu = self.swish(p4_bu)
        p4_bu = self.conv_bu['conv_p4_bu'](p4_bu)

        p4_d = self.downsample_func(p4_bu)
        p5_bu = self.wadd_bu['wadd_p5_bu']([p5_in_2, p5_td, p4_d])
        p5_bu = self.swish(p5_bu)
        p5_bu = self.conv_bu['conv_p5_bu'](p5_bu)

        p5_d = self.downsample_func(p5_bu)
        p6_bu = self.wadd_bu['wadd_p6_bu']([p6_in, p6_td, p5_d])
        p6_bu = self.swish(p6_bu)
        p6_bu = self.conv_bu['conv_p6_bu'](p6_bu)

        p6_d = self.downsample_func(p6_bu)
        p7_bu = self.wadd_bu['wadd_p7_bu']([p7_in, p6_d])
        p7_bu = self.swish(p7_bu)
        p7_bu = self.conv_bu['conv_p7_bu'](p7_bu)

        return p3_td, p4_bu, p5_bu, p6_bu, p7_bu
