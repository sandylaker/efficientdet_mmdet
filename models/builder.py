from models.backbones import EfficientNetBackBone
from models.detectors import EfficientDet
from models.necks import BiFPN
from mmdet.models import BACKBONES, NECKS, DETECTORS
from mmdet.models.builder import build


BACKBONES.register_module(name='EfficientNet', module=EfficientNetBackBone)
NECKS.register_module(name='BiFPN', module=BiFPN)
DETECTORS.register_module(name='EfficientDet', module=EfficientDet)


def build_detector(cfg: dict, train_cfg: dict = None, test_cfg: dict = None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))

