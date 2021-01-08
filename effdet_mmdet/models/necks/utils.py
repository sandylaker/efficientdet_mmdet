import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from copy import deepcopy


class DepthwiseSeparableConvModule(nn.Module):
    """Modified depthwise separable convolution module. This allows for setting no bias
    to the depthwise convolution but bias to the pointwise convolution.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if `norm_cfg` and `act_cfg` are specified.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d. Default: 1.
        padding (int or tuple[int]): Same as nn.Conv2d. Default: 0.
        dilation (int or tuple[int]): Same as nn.Conv2d. Default: 1.
        norm_cfg (dict): Default norm config for both depthwise ConvModule and
            pointwise ConvModule. Default: None.
        act_cfg (dict): Default activation config for both depthwise ConvModule
            and pointwise ConvModule. Default: dict(type='ReLU').
        dw_norm_cfg (dict): Norm config of depthwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        dw_act_cfg (dict): Activation config of depthwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        pw_norm_cfg (dict): Norm config of pointwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        pw_act_cfg (dict): Activation config of pointwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        kwargs (optional): Other shared arguments for depthwise and pointwise
            ConvModule. See ConvModule for ref.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 dw_norm_cfg='default',
                 dw_act_cfg='default',
                 pw_norm_cfg='default',
                 pw_act_cfg='default',
                 **kwargs):
        super(DepthwiseSeparableConvModule, self).__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'

        # if norm/activation config of depthwise/pointwise ConvModule is not
        # specified, use default config.
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        dw_kwargs = deepcopy(kwargs)
        dw_kwargs.update({'bias': False})
        # depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
            **dw_kwargs)

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
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
        self.weights = nn.Parameter(torch.Tensor(self.num_inputs,).fill_(init_val))

    def forward(self, x):
        assert len(x) == self.num_inputs
        weights = self.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        x = reduce_weighted_sum(x, weights)
        return x