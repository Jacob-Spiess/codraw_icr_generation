#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script that runs the complete training, validation and test. The evaluation is performed in the JupyterNotebook.
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
from icr.icrgenerator import ICRModel, ICRModel1, ICRModel2, ICRModelBart
from icr.aux import write_model_outputs_to_files


print('\n---------- Running iCR experiment ----------\n')

pl.seed_everything(config["training"]["random_seed"])
torch.use_deterministic_algorithms(True, warn_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = Vocabulary(config["data"]["codraw_path"])
dm = CoDrawDataModule(data_config=config["data"], batch_size=config["generation"]["batch_size"], vocabulary=vocab)

model = ICRModel(vocab, config["model"]).to(device) #Bart

print('\n---------- Initiliaze Trainer ----------\n')

checkpoint_callback = ModelCheckpoint(save_top_k=1, verbose=True, monitor='val_loss',mode='min')
early_stopper = EarlyStopping(monitor='val_loss', mode='min', patience=1, min_delta=0.001)
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

print('\n---------- Start evaluating ----------\n')
model.eval()
torch.no_grad()

val_data = dm.val_dataloader()
target_file = "targets.txt"

#can change based on vocabulary settings
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

output_paths = {
    "reply": config["paths"]["outputs_path"]+"icr_predictions.txt",
    "outputs": config["paths"]["outputs_path"]+"out_predictions.txt",
    "n": config["paths"]["outputs_path"]+"number_predictions.txt",
    "c": config["paths"]["outputs_path"]+"clipart_predictions.txt",
    "t": config["paths"]["outputs_path"]+"topic_predictions.txt",
    "m": config["paths"]["outputs_path"]+"mood_predictions.txt",
}

write_model_outputs_to_files(model, val_data, output_paths, use_sampling=True, top_k=5, top_p=0.95, temperature = 1)  #, bart=True

print("Outputs safed!\n")
print("First 5 Outputs:\n")
# Print only the first 5 outputs from the file
with open(output_paths["reply"], "r") as file:
    lines = [line.strip() for line in file.readlines()]
    print(lines[0:5])

