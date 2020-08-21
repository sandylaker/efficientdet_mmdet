import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class SeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 inplace=False):
        super(SeparableConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.depthwise_conv = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,   # noqa
            inplace=inplace)

        self.pointwise_conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=inplace)

    def forward(self, x):
        x = self.depthwise_conv(x, activate=False, norm=False)
        x = self.pointwise_conv(x)
        return x


def reduce_weighted_sum(inputs, weights):
    inputs = torch.stack([inputs[i] * weights[i] for i in range(len(weights))])
    return torch.sum(inputs, 0)


class WeightedAdd(nn.Module):
    def __init__(self, num_inputs, eps=1e-4):
        super(WeightedAdd, self).__init__()
        self.num_inputs = num_inputs
        self.eps = eps
        self.relu = nn.ReLU()

        init_val = 1.0 / self.num_inputs
        self.weights = nn.Parameter(torch.Tensor(2,).fill_(init_val))

    def forward(self, x):
        assert len(x) == self.num_inputs
        weights = self.relu(self.weights)
        weights /= weights.sum() + self.eps
        x = reduce_weighted_sum(x, weights)
        return x