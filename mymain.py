#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script that runs the complete training, validation and test.
"""

import warnings

import comet_ml
import pytorch_lightning as pl
import torch
#from torch.utils.data import DataLoader, Dataset
#from pytorch_lightning.loggers import CometLogger
#from pytorch_lightning.callbacks import (
#    ModelCheckpoint, LearningRateMonitor, EarlyStopping)

#from icr import aux
#from icr.constants import COMET_LOG_PATH, SPLITS
from icr.config import config
from icr.vocabulary import Vocabulary
from icr.dataloader import CodrawData
from icr.datamodel import CoDrawDataModule
from icr.icrgenerator import iCRGeneration, ICRModel, ICRModel1



print('\n---------- Running iCR experiment ----------\n')

#pl.seed_everything(config["training"]["random_seed"])
#torch.use_deterministic_algorithms(True, warn_only=True)

vocab = Vocabulary(config["data"]["codraw_path"])
dm = CoDrawDataModule(data_config=config["data"], batch_size=config["generation"]["batch_size"], vocabulary=vocab)

model = ICRModel1(vocab, config["model"])

#model = iCRGeneration(config["generation"], config["model"])

print('\n---------- Initiliaze Trainer ----------\n')
trainer = pl.Trainer(
    max_epochs=config["training"]["n_epochs"], 
    accelerator=config["training"]["device"],
    devices=config["training"]["gpu"])

print('\n---------- Start fitting ----------\n')
trainer.fit(model, dm)

print('\n---------- Start testing ----------\n')
trainer.test(model, datamodule=dm)

print('\n---------- Start evaluating ----------\n')
model.eval()

#y_hat = model(x)
#print(y_hat)
