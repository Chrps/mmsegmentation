# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class PascalMosDataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    CLASSES = ('background', 'CleanRoof', 'AffectedRoof')

    PALETTE = [[0,0,0], [128,0,0], [0,128,0]]

    def __init__(self, split, **kwargs):
        super(PascalMosDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None