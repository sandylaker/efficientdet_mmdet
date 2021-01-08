from mmdet.models import SingleStageDetector


class EfficientDet(SingleStageDetector):
    # maps scale to architecture settings
    # (Width, Depth, Depth of bbox/class head, input channels list of BiFPN)
    # where the input channels are from p3, p4, p5 of backbone
    param_dict = {
        0: (64, 3, 3, [40, 112, 320]),
        1: (88, 4, 3, [40, 112, 320]),
        2: (112, 5, 3, [48, 120, 352]),
        3: (160, 6, 4, [48, 136, 384]),
        4: (224, 7, 4, [56, 160, 448]),
        5: (228, 7, 4, [64, 176, 512]),
        6: (384, 8, 5, [72, 200, 576]),
        7: (384, 8, 5, [80, 224, 640]),
    }

    def __init__(self,
                 backbone:dict,
                 neck: dict,
                 bbox_head: dict,
                 scale: int = 1,
                 train_cfg:dict = None,
                 test_cfg: dict = None,
                 pretrained=None):
        assert scale in range(0, 8)
        bifpn_width, bifpn_depth, head_depth, bifpn_channels_list = self.param_dict[scale]

        backbone.update({'scale': scale})
        neck.update({'in_channels_list': bifpn_channels_list,
                     'out_channels': bifpn_width,
                     'stack': bifpn_depth})
        bbox_head.update({'stacked_convs': head_depth,
                          'in_channels': bifpn_width,
                          'feat_channels': bifpn_width})

        super(EfficientDet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                           test_cfg, pretrained)