#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script that runs the complete training, validation and test.
"""

import warnings
import os
import comet_ml
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
#from torch.utils.data import DataLoader, Dataset
#from pytorch_lightning.loggers import CometLogger
#from pytorch_lightning.callbacks import (
#    ModelCheckpoint, LearningRateMonitor, EarlyStopping)

#from icr import aux
#from icr.constants import COMET_LOG_PATH, SPLITS
from icr.constants import EOS
from icr.config import config
from icr.vocabulary import Vocabulary
from icr.dataloader import CodrawData
from icr.datamodel import CoDrawDataModule
from icr.icrgenerator import ICRModel1



print('\n---------- Running iCR experiment ----------\n')

#pl.seed_everything(config["training"]["random_seed"])
#torch.use_deterministic_algorithms(True, warn_only=True)

vocab = Vocabulary(config["data"]["codraw_path"])
dm = CoDrawDataModule(data_config=config["data"], batch_size=config["generation"]["batch_size"], vocabulary=vocab)

CKPT_PATH = "./outputs/lightning_logs/version_18/checkpoints/epoch=13-step=3066.ckpt"
#CKPT_PATH = './outputs/lightning_logs/version_16/checkpoints/epoch=15-step=3504.ckpt'

model = ICRModel1.load_from_checkpoint(CKPT_PATH)

#print(model.learning_rate)
# prints the learning_rate you used in this checkpoint

model.eval()

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

trainer = pl.Trainer(
    max_epochs=config["training"]["n_epochs"], 
    accelerator=config["training"]["device"],
    devices=config["training"]["gpu"],
    default_root_dir=config["paths"]["outputs_path"],
    callbacks=[checkpoint_callback])


trainer.test(model, datamodule=dm)


val_data = dm.val_dataloader()
output_file = "predictions.txt"
target_file = "targets.txt"

if os.path.isfile(config["paths"]["outputs_path"]+target_file) != True:
    with open(config["paths"]["outputs_path"]+target_file, "w") as file:
        for batch_idx, batch in enumerate(val_data):
            for reply in batch["drawer_reply_tokenized"]:
                x_target = []
                for i in reply:
                    x_target.append(vocab.itos[i.item()])
                    if vocab.itos[i.item()] == EOS:
                        break
                file.write(' '.join(x_target[1:-1]) + '\n')
            
with open(config["paths"]["outputs_path"]+output_file, "w") as file:
    for batch_idx, batch in enumerate(val_data):
        for d in batch["dialogue"]:
            dialogue = d.unsqueeze(0)
            predictions = model.reply(dialogue)
            file.write(' '.join(predictions[:-1]) + '\n')

print("Outputs safed!\n")
print("First 5 Outputs:\n")
# Print only the first 5 outputs from the file
with open(config["paths"]["outputs_path"]+output_file, "r") as file:
    lines = [line.strip() for line in file.readlines()]
    print(lines[0:5])