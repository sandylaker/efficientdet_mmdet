from mmdet.datasets import DATASETS, ConcatDataset, RepeatDataset, ClassBalancedDataset
import os.path as osp
ROOT = osp.dirname(osp.dirname(__file__))
import sys
sys.path.append(ROOT)
from mmcv.utils import Config, build_from_cfg
from .voc import MyVOCDataset
import copy


DATASETS.register_module('MyVOCDataset', module=MyVOCDataset)


def _concat_dataset(cfg, default_args=None):
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset



if __name__ == '__main__':
    from mmdet.datasets import build_dataloader
    cfg = Config.fromfile(osp.join(ROOT, 'configs/efficientdet.py'))
    train_set = build_dataset(cfg.data.train, dict(test_mode=False))
    train_loader = build_dataloader(
        train_set,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
    )
    for i, data in enumerate(train_loader):
        if i >= 1:
            break
        print('Batch {}'.format(i + 1))
        print(data.keys())
