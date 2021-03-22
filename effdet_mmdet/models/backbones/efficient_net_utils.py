import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Swish


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_planes: int, se_planes: int):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(in_planes, se_planes, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(se_planes, in_planes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x


class MBConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 kernel_size: int,
                 stride: int,
                 expand_rate: float = 1.0,
                 se_rate: float = 0.25,
                 drop_connect_rate: float = 0.2,
                 norm_cfg=dict(type='SyncBN', momentum=0.01, eps=1e-3),
                 act_cfg=dict(type='Swish')):
        super(MBConv, self).__init__()

        expand_planes = int(in_planes * expand_rate)
        se_planes = max(1, int(in_planes * se_rate))

        self.expansion_conv = None
        if expand_rate > 1.0:
            self.expansion_conv = ConvModule(
                in_channels=in_planes,
                out_channels=expand_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            in_planes = expand_planes

        self.depthwise_conv = ConvModule(
            in_channels=in_planes,
            out_channels=expand_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expand_planes,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.squeeze_excitation = SqueezeExcitation(expand_planes, se_planes)

        self.project_conv = ConvModule(
            in_channels=expand_planes,
            out_channels=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)   # noqa

        self.with_skip = stride == 1
        self.drop_connect_rate = torch.tensor(drop_connect_rate, requires_grad=False)

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate    # noqa
        drop_mask = torch.rand(x.shape[0], 1, 1, 1) + keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / keep_prob

    def forward(self, x):
        z = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)

        # Add identity skip
        if x.shape == z.shape and self.with_skip:
            if self.training and self.drop_connect_rate is not None:
                x = self._drop_connect(x)
            x += z
        return x
