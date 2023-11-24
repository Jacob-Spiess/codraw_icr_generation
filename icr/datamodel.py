#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the DataModule for the ecperiment.
"""

from typing import Dict, Tuple, List
import pytorch_lightning as pl
from icr import aux
from icr.constants import COMET_LOG_PATH, SPLITS
from icr.dataloader import CodrawData
from torch.utils.data import DataLoader

class CoDrawDataModule(pl.LightningDataModule):
    def __init__(self, data_config: Dict[str, int], batch_size: int, vocabulary):
        
        super().__init__()
        self.batch_size=batch_size
        clipmap = aux.define_cliparts(data_config["dont_merge_persons"])
        self.datasets = {split: CodrawData(split, clipmap, vocabulary, **data_config)
                    for split in SPLITS}

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False)
    