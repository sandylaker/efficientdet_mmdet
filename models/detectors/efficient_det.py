from mmdet.models import SingleStageDetector


class EfficientDet(SingleStageDetector):
    def __init__(self,
                 backbone:dict,
                 neck: dict,
                 bbox_head: dict,
                 train_cfg:dict = None,
                 test_cfg: dict = None,
                 pretrained=None):
        super(EfficientDet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                           test_cfg, pretrained)