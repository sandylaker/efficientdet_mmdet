from .backbones import EfficientNetBackBone
from .detectors import EfficientDet
from .necks import BiFPN
from .heads import EfficientHead
from mmdet.models import BACKBONES, NECKS, HEADS, DETECTORS
from mmdet.models.builder import build


BACKBONES.register_module(name='EfficientNet', module=EfficientNetBackBone)
NECKS.register_module(name='BiFPN', module=BiFPN)
HEADS.register_module(name='EfficientHead', module=EfficientHead)

DETECTORS.register_module(name='EfficientDet', module=EfficientDet)


def build_detector(cfg: dict, train_cfg: dict = None, test_cfg: dict = None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))

