import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule, xavier_init
from models.necks.utils import SeparableConv
from models.backbones.efficient_net_utils import Swish


class BiFPN(nn.Module):
    def __init__(self,
                 in_channels: list,
                 out_channels: int,
                 num_outs: int,
                 start_level:int = 0,
                 end_level: int = -1,
                 stack: int = 1,
                 add_extra_convs: str = None,
                 relu_before_extra_convs: bool = False,
                 no_norm_on_lateral: bool = False,
                 conv_cfg: dict = None,
                 norm_cfg: dict = dict(type='BN', momentum=0.01, eps=1e-3),
                 act_cfg: dict = None,
                 upsample_cfg: dict = dict(mode='nearest')):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        # add channels for two down-sampled inputs (p6, p7)
        in_channels.extend([out_channels, out_channels])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg
        self.stack = stack
        self.swish = Swish()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        if add_extra_convs is not None:
            assert isinstance(add_extra_convs, str)
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        else:
            add_extra_convs = False
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        # extra convolution layers
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList()

        # get p6 and p7
        # in_channels index should be -3, since the last two elements are equal to out_channels
        self.input_conv = ConvModule(
            in_channels=in_channels[-3],
            out_channels=out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)
        self.input_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.input_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # build laterals
        # skip the last two level, for which the channels are adjusted with input_conv and pooling
        for i in range(self.start_level, self.backbone_end_level-2):
            l_conv = ConvModule(
                in_channels=in_channels[i],
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=None,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for ii in range(stack):
            self.stack_bifpn_convs.append(BiFPNModule(
                channels=out_channels,
                levels=self.backbone_end_level - self.start_level,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                upsample_cfg=upsample_cfg))

        # add extra conv layers
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = list(inputs)
        p6 = self.input_pool_1(self.input_conv(inputs[-1]))
        p7 = self.input_pool_2(p6)
        inputs.extend([p6, p7])

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # append the last two levels
        laterals.extend([p6, p7])

        # build top-down and bottom-up with stack
        used_backbone_levels = len(laterals)
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals

        # add extra levels
        if self.num_outs > len(outs):
            # use max-pool to get more levels on top of outputs
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps or laterals or outputs
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[0](extra_source))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class BiFPNModule(nn.Module):
    def __init__(self,
                 channels: int,
                 levels: int,
                 init:float = 0.5,
                 conv_cfg:dict = None,
                 norm_cfg:dict = None,
                 act_cfg:dict = None,
                 upsample_cfg: dict = dict(mode='nearest'),
                 eps:float = 1e-4):
        super(BiFPNModule, self).__init__()
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.eps = eps
        self.levels = levels
        self.upsample_cfg = upsample_cfg
        self.bifpn_convs = nn.ModuleList()
        self.swish = Swish()

        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        self.relu2 = nn.ReLU()

        for jj in range(2):
            for i in range(self.levels - 1):
                fpn_conv = SeparableConv(
                    in_channels=channels,
                    out_channels=channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                # the order top-down and then bottom-up
                self.bifpn_convs.append(fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.levels
        levels = self.levels

        w1 = self.relu1(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.eps
        w2 = self.relu2(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.eps

        idx_bifpn = 0
        pathtd = inputs
        inputs_clone = []
        for in_tensor in inputs:
            inputs_clone.append(in_tensor.clone())

        # build top-down
        for i in range(levels - 1, 0, -1):
            norm_factor = w1[0, i-1] + w1[1, i-1] + self.eps

            if 'scale_factor' in self.upsample_cfg:
                interpolated_feat = F.interpolate(pathtd[i], **self.upsample_cfg)
            else:
                prev_shape = inputs[i-1].shape[2:]
                interpolated_feat = F.interpolate(pathtd[i], size=prev_shape, **self.upsample_cfg)
            pathtd[i-1] = (w1[0, i-1] * pathtd[i-1] + w1[1, i-1] * interpolated_feat) / norm_factor
            pathtd[i-1] = self.bifpn_convs[idx_bifpn](self.swish(pathtd[i-1]))
            idx_bifpn += 1
        # build bottom-up
        for i in range(0, levels - 2, 1):
            norm_factor = w2[0, i] + w2[1, i] + w2[2, i] + self.eps

            next_shape = inputs[i+1].shape[2:]
            max_pooled_feat = F.adaptive_max_pool2d(pathtd[i], output_size=next_shape)
            pathtd[i+1] = (w2[0, i] * pathtd[i+1] + w2[1, i] * max_pooled_feat +
                           w2[2, i] * inputs_clone[i+1]) / norm_factor
            pathtd[i+1] = self.bifpn_convs[idx_bifpn](self.swish(pathtd[i+1]))
            idx_bifpn += 1

        pathtd[levels-1] = (w1[0, levels-1] * pathtd[levels-1] + w1[1, levels-1] * F.max_pool2d(
            pathtd[levels-2], kernel_size=2)) / (w1[0, levels-1] + w1[1, levels-1] + self.eps)
        pathtd[levels-1] = self.bifpn_convs[idx_bifpn](self.swish(pathtd[levels-1]))
        return pathtd


if __name__ == '__main__':
    feats = [
        torch.rand([4, 40, 128, 128]),
        torch.rand([4, 112, 64, 64]),
        torch.rand([4, 320, 32, 32])
    ]
    neck = BiFPN(
        in_channels=[40, 112, 320],
        out_channels=64,
        num_outs=5)
    feats = neck(feats)
    for f in feats:
        print(f.shape)
