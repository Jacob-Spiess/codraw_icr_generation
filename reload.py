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

from icr.constants import EOS
from icr.config import config
from icr.vocabulary import Vocabulary
from icr.dataloader import CodrawData
from icr.datamodel import CoDrawDataModule
from icr.icrgenerator import ICRModel, ICRModel1
from icr.aux import write_model_outputs_to_files



print('\n---------- Running iCR experiment ----------\n')

#pl.seed_everything(config["training"]["random_seed"])
#torch.use_deterministic_algorithms(True, warn_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = Vocabulary(config["data"]["codraw_path"])
dm = CoDrawDataModule(data_config=config["data"], batch_size=config["generation"]["batch_size"], vocabulary=vocab)

#'comet-logs/codraw-icr-generation/7cc309e780ac4149bd6a514a231f0e2a/checkpoints/epoch=18-step=4161.ckpt' #best Transformer
#'comet-logs/codraw-icr-generation/a8fbfbb69dbe47efb065d6e01445572b/checkpoints/epoch=39-step=8760.ckpt' #best LSTM secret_cougar_1115
#'comet-logs/codraw-icr-generation/468af100307a488bb9193aed3d1f1896/checkpoints/epoch=14-step=3285.ckpt' #best Attention diverse_stable_1301
#'comet-logs/codraw-icr-generation/4e2691240c9648cd876104d0e83c6c87/checkpoints/epoch=39-step=8760.ckpt' #Baseline LSTM no classifiers
CKPT_PATH = 'comet-logs/codraw-icr-generation/780a640516744ded8c7c6cd4ce3269c3/checkpoints/epoch=39-step=8760.ckpt'
#CKPT_PATH = 'comet-logs/codraw-icr-generation/969472473b404df49daa89be78ed0106/checkpoints/epoch=9-step=2190.ckpt'
#CKPT_PATH = 'outputs/lightning_logs/version_85/checkpoints/epoch=3-step=657.ckpt'
#CKPT_PATH = './outputs/lightning_logs/version_16/checkpoints/epoch=15-step=3504.ckpt'

model = ICRModel.load_from_checkpoint(CKPT_PATH)
model.to(device)

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

write_model_outputs_to_files(model, val_data, output_paths, use_sampling=True, top_k=50, top_p=0.60, temperature = 2, 
                             write_additional_outputs=False)

print("Outputs safed!\n")
print("First 5 Outputs:\n")
# Print only the first 5 outputs from the file
with open(output_paths["reply"], "r") as file:
    lines = [line.strip() for line in file.readlines()]
    print(lines[0:5])
