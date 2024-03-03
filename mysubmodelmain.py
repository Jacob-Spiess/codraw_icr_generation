#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script that runs the complete training, validation and test for model versions that just focus on the training of single intermediate decoders to later load them into the ICRpretrainedModel.
"""
import warnings
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
import comet_ml
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from icr.constants import EOS, COMET_LOG_PATH
from icr.config import config
from icr.vocabulary import Vocabulary
from icr.dataloader import CodrawData
from icr.datamodel import CoDrawDataModule
from icr.submodels import NModel, CModel, TModel, MModel
from icr.aux import write_model_outputs_to_files


print('\n---------- Running iCR experiment ----------\n')

pl.seed_everything(config["training"]["random_seed"])
torch.use_deterministic_algorithms(True, warn_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_num_threads(1)

vocab = Vocabulary(config["data"]["codraw_path"])
dm = CoDrawDataModule(data_config=config["data"], batch_size=config["generation"]["batch_size"], vocabulary=vocab)

model = MModel(vocab, config["model"]).to(device)

print('\n---------- Initiliaze Trainer ----------\n')

checkpoint_callback = ModelCheckpoint(save_top_k=1, verbose=True, monitor='val_loss_m', mode='min') #val_loss_n, val_loss_c, val_loss_t
early_stopper = EarlyStopping(monitor='val_loss_m', mode='min', patience=2, min_delta=0.001) #val_loss_n ,val_loss_c, val_loss_t, 
cometlog = CometLogger(api_key=config["comet"]["comet_key"], workspace=config["comet"]["comet_workspace"],
                     save_dir=COMET_LOG_PATH, project_name=config["comet"]["comet_project"])
lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum=True)

trainer = pl.Trainer(
    max_epochs = config["training"]["n_epochs"], 
    accelerator = config["training"]["device"],
    devices = config["training"]["gpu"],
    default_root_dir = config["paths"]["outputs_path"],
    logger = cometlog,
    gradient_clip_val = config["training"]["clip"] if config["training"]["clip"] != 0 else None,
    reload_dataloaders_every_n_epochs = config["training"]["n_reload_data"],
    accumulate_grad_batches = config["training"]["n_grad_accumulate"],
    callbacks = [lr_monitor, checkpoint_callback, early_stopper])

print('\n---------- Start fitting ----------\n')
trainer.fit(model, dm)

print('\n---------- Start testing ----------\n')
trainer.test(model, datamodule=dm)

print('\n---------- Save the Model ----------\n')
torch.save(model.state_dict(), './savedsubmodels/MModel.pth')