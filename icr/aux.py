#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions used in various scripts.
"""

import json
import os
import re
import warnings
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
#from pytorch_lightning.loggers import Logger
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

from icr import constants
from icr.structs.dataconf import file2obj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_model_outputs_to_files(model, val_data, output_paths, use_sampling, top_k, top_p, temperature, bart=False):
    """
    Writes the outputs of the model to specified files. Sampling parameters are defined and considered here for the inference 
    process. There is a flag for models with a BART decoder, since it works with a specific tokenizer, 
    which alters the tokenhandling slightly.
    """
    # Open all files at once
    with open(output_paths["reply"], "w") as file1, \
         open(output_paths["outputs"], "w") as file2, \
         open(output_paths["n"], "w") as file3, \
         open(output_paths["c"], "w") as file4, \
         open(output_paths["t"], "w") as file5, \
         open(output_paths["m"], "w") as file6:
        
        outputs, n, c, t, m = [], [], [], [], []
        
        for batch_idx, batch in enumerate(val_data):
            for dialogue, scene_before, scene_after in zip(batch["dialogue"], batch["scene_before"], batch["scene_after"]):
                dialogue = dialogue.unsqueeze(0).to(device)
                scene_before = scene_before.unsqueeze(0).to(device)
                scene_after = scene_after.unsqueeze(0).to(device)
                model.to(device)
                
                predictions = model.reply(dialogue, scene_before, scene_after, use_sampling, top_k, top_p, temperature)
                
                if bart:
                    file1.write(predictions[0].replace("</s>","") + '\n')
                else:
                    file1.write(' '.join(predictions[0][:-1]) + '\n')
                outputs.append(predictions[1].tolist())
                n.append(predictions[2].tolist())
                c.append(predictions[3].tolist())
                t.append(predictions[4].tolist())
                m.append(predictions[5].tolist())

        json.dump(outputs, file2)
        json.dump(n, file3)
        json.dump(c, file4)
        json.dump(t, file5)
        json.dump(m, file6)

'''
def write_model_outputs_to_files(model, val_data, output_paths, use_sampling, top_k, top_p, temperature, bart=False):
    """
    Writes the outputs of the model to specified files. Sampling parameters are defined and considered here for the inference 
    process. There is a flag for models with a BART decoder, since it works with a specific tokenizer, 
    which alters the token handling slightly.
    """
    outputs, n, c, t, m = [], [], [], [], []

    with open(output_paths["reply"], "w") as file1, \
         open(output_paths["outputs"], "w") as file2, \
         open(output_paths["n"], "w") as file3, \
         open(output_paths["c"], "w") as file4, \
         open(output_paths["t"], "w") as file5, \
         open(output_paths["m"], "w") as file6:

        for batch_idx, batch in enumerate(val_data):
            dialogues = batch["dialogue"].to(device)
            scenes_before = batch["scene_before"].to(device)
            scenes_after = batch["scene_after"].to(device)

            model.to(device)
            with torch.no_grad():
                predictions = model.reply(dialogues, scenes_before, scenes_after, use_sampling, top_k, top_p, temperature)

            if bart:
                file1.write(predictions[0].replace("</s>", "") + '\n')
            else:
                file1.write(' '.join(predictions[0][:-1]) + '\n')

            outputs.append(predictions[1])
            n.append(predictions[2])
            c.append(predictions[3])
            t.append(predictions[4])
            m.append(predictions[5])

        json.dump(outputs, file2)
        json.dump(n, file3)
        json.dump(c, file4)
        json.dump(t, file5)
        json.dump(m, file6)
'''
        
def load_partial_state_dict(model, state_dict, prefix):
    """Writes the saved model parameters into the new given model, which is just a subpart of the encoder-decoder architecture."""
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name.startswith(prefix):
            name = name[len(prefix):]  # Remove the prefix
            own_state[name].copy_(param)

            
def filter_config(params: Namespace, keys: list) -> Namespace:
    """Filter the parameters by keys and return a Namespace with the subset."""
    subset = {key: value for key, value in vars(params).items() if key in keys}
    return Namespace(**subset)


def split_config(params: Namespace) -> List[Namespace]:
    """Split arguments into groups for each component of the experiment."""
    data_config = filter_config(params, constants.DATA_CONFIG)
    comet_config = filter_config(params, constants.COMET_CONFIG)
    train_config = filter_config(params, constants.TRAINER_CONFIG)
    model_config = filter_config(params, constants.MODEL_CONFIG)
    exp_config = filter_config(params, constants.EXPERIMENT_CONFIG)
    return data_config, comet_config, train_config, model_config, exp_config


def split_batch(batch, bart=False): 
    """Split batch data into inputs for the encoder"""
    if bart:
        drawer_sequence = batch["drawer_reply_tokenized_bart"][:, :-1]
        targets =  batch["drawer_reply_tokenized_bart"][:,1: ]
    else:
        drawer_sequence = batch["drawer_reply_tokenized"][:, :-1]
        targets =  batch["drawer_reply_tokenized"][:,1: ]
    dialogue = batch["dialogue"]
    clip = batch["icr_clip_label"]
    mood = batch["icr_mood"]
    num_clip = batch["icr_num_clip"]
    topic = batch["icr_topic"]
    scene_before = batch["scene_before"]
    scene_after = batch["scene_after"]
    return drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic

def compute_accuracy_text(outputs, targets, pad_idx):
    """Calculate accuracy for the predicted tokens matching the target, while ignoring the padding"""
    #with padding:
    #acc = (outputs.view(-1, outputs.size(-1)).argmax(dim=1) == targets.reshape(-1)).sum().item() / len(targets.reshape(-1))
    
    outputs_flat = outputs.view(-1, outputs.size(-1))
    targets_flat = targets.reshape(-1)
    predictions = outputs_flat.argmax(dim=1)

    non_pad_mask = targets_flat != pad_idx

    correct_non_pad = (predictions == targets_flat) & non_pad_mask
    accuracy = correct_non_pad.sum().item() / non_pad_mask.sum().item()

    return accuracy


def compute_accuracy_single_label(preds, y):
    """Function to calculate the accuracy for a single label model."""
    preds = torch.argmax(preds, dim=1)
    correct = (preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def compute_accuracy_multi_label(preds, y):
    """Function to calculate the accuracy for a multi label model."""
    preds = preds > 0.5 # Threshold predictions at 0.5
    correct = (preds == y).float()
    acc = correct.sum() / (correct.numel())
    return acc


def check_params_consistency(params):
    """Raises error if invalid combinations of hyperparameters is detected."""

    # either pure ablation only with state, or at least one source of input
    if not params.random_baseline:
        assert (not params.no_instruction
                or params.use_scene_after
                or params.use_scene_before), "At least one input is needed!"
    if params.random_baseline:
        assert params.actions_for_icr == 'none', "No input allowed for random base."
        assert params.no_instruction, "No input allowed for random base."
        assert not params.use_scene_after, "No input allowed for random base."
        assert not params.use_scene_before, "No input allowed for random base."

    # at least one prediction
    assert (not params.dont_make_actions
            or params.predict_icrs_clipart
            or params.predict_icrs_turn), "At least one prediction is needed!"

    assert not (params.predict_icrs_clipart
                and params.predict_icrs_turn), "Only one type of iCR prediction is possible!"

    if params.use_scene_after and not params.dont_make_actions:
        warn_msg = 'Actions will be *detected* from scenes, not *predicted*!'
        warnings.warn(warn_msg)

    # logits can only be used is actions are being predicted
    if params.actions_for_icr == 'logits':
        assert not params.dont_make_actions

    # set boundaries of the CoDraw game score
    assert 0 <= params.score_threshold <= 5

    if params.unfreeze_resnet or params.dont_preprocess_scenes:
        assert params.use_scene_before or params.use_scene_after

'''
def log_all(logger: Logger, params: Namespace, datasets: Dict[str, Dataset],
            clipmap: Dict[str, int], monitored_metric: str) -> Path:
    """Log all hyperparameters and metadata to comet and to local folder."""

    # create directory where experiment results and metadata will be saved
    path_name = Path(f'{params.outputs_path}{logger.version}')
    os.mkdir(path_name)

    # log all to comet
    params_dic = {k: v for k, v in vars(params).items() if '_key' not in k}
    logger.experiment.log_parameters(params_dic)
    logger.experiment.log_code('main.py')
    logger.experiment.log_code(folder='icr/')
    logger.experiment.log_others(datasets['train'].stats)
    logger.experiment.log_others(datasets['val'].stats)
    logger.experiment.log_others(datasets['test'].stats)
    logger.experiment.log_other('monitored', monitored_metric)
    if params.comet_tag:
        logger.experiment.add_tag(params.comet_tag)

    # also log hyperparameter configuration and metadata to local
    for split in constants.SPLITS:
        with open(path_name / f'{split}_stats.json', 'w') as file:
            json.dump(datasets['train'].stats, file)

    with open(path_name / 'config.json', 'w') as file:
        json.dump(params_dic, file)

    with open(path_name / 'clipmap.json', 'w') as file:
        json.dump(clipmap, file)

    with open(path_name / 'meta.txt', 'w') as file:
        file.write(f'Comet name: {logger.experiment.get_name()}\n')
        file.write(f'Monitored: {monitored_metric}\n')
        file.write(f'Tag: {params.comet_tag}')

    datasets['val'].save_structure(path_name)
    datasets['test'].save_structure(path_name)
    print('\n')
    return path_name


def log_final_state(logger: Logger, log_dir: Path, ckpt_name: str) -> None:
    """Log the best epoch to the the local directory."""
    begin = re.search('epoch=', ckpt_name).span()[1]
    end = re.search('.ckpt', ckpt_name).span()[0]
    epoch = int(ckpt_name[begin: end])

    with open(log_dir / 'meta.txt', 'a') as file:
        file.write(f'Best epoch: {epoch}')
    logger.experiment.log_other('best_epoch', epoch)
    logger.experiment.log_asset_folder(log_dir)
    logger.experiment.log_model('best-model.ckpt', ckpt_name)
'''
def encode_classes(column):
    """One hot encode a specific column in the annotation table"""
    column = column.fillna(column.name+"_nan")

    categories_list = [entry.split(',') for entry in column]

    mlb = MultiLabelBinarizer()
    encoded_data = mlb.fit_transform(categories_list)
    encoded_df = pd.DataFrame(encoded_data, columns=mlb.classes_)

    encoded_df.columns = [col.replace(' ', '_') for col in encoded_df.columns]
    encoded_df.columns = [col.replace('-', '_') for col in encoded_df.columns]

    return encoded_df


def get_mentioned_cliparts(row: pd.Series) -> List[str]:
    """Make list of mentioned cliparts in an iCR."""
    mentioned = [row.clipart_1, row.clipart_2, row.clipart_3, row.clipart_4,
                 row.clipart_5]
    return [clip.replace('_', ' ') for clip in mentioned if not pd.isna(clip)]


def get_question_mood(row: pd.Series) -> List[str]:
    """Make list of one hot encoded mood in an iCR."""
    return torch.Tensor([row.mood_nan, row.alternative_question, row.declarative, row.imperative, 
                         row.other, row.polar_question, row.wh_question])


def get_icr_topic(row: pd.Series) -> List[str]:
    """Make list of one hot encoded topic of an iCR."""
    return torch.Tensor([row.position, row.size_, row.direction, row.relation_to_other_cliparts, row.disambig_object, row.disambig_person])


def get_number_cliparts(row: pd.Series) -> List[str]:
    """Make list of one hot number of talked about cliparts in an iCR."""
    return torch.Tensor([row.clipart_nan, row.two, row.many, row.one, row.unknown])


def parse_id(name: str) -> int:
    """Extract the id from the {train, val, test}_id game name as int."""
    return int(name.split('_')[1])


def get_pose_face(name: str) -> Tuple[int, int]:
    """Get an id for the pose and face, if applicable, or return a dummy id."""
    if 'boy' in name or 'girl' in name:
        _, face, pose = name.split()
        return [constants.POSES[pose], constants.FACES[face]]
    return constants.POSES['n/a'], constants.POSES['n/a']


def get_attributes(cliplist: Dict[str, Any], attribute: str) -> Tensor:
    """Return a tensor with the attribute value for all cliparts in scene."""
    return torch.tensor([clip[attribute] for clip in cliplist])


def define_cliparts(dont_merge_persons: bool) -> Dict[str, int]:
    """Create a mapping from cliparts to integers."""
    if dont_merge_persons:
        clipmap = {clip: i for i, clip in enumerate(file2obj.values())}
    else:
        # merge all boys into one and all girls into generic categories
        short_clips = [x for x in file2obj.values()
                       if 'boy' not in x and 'girl' not in x]
        short_clips += ['boy', 'girl']
        clipmap = {clip: i for i, clip in enumerate(short_clips)}
    return clipmap


def mask_pads(x: Tensor, pe: Tensor) -> Tensor:
    """Mask positional encoding to zero in pad tokens."""
    # turn all padding elements to zero, so that these tokens are completely
    # disregarded in the attention mechanism
    mask = (x != 0).all(dim=2).int().unsqueeze(2)
    full_mask = mask.expand(-1, -1, x.shape[2])
    return torch.mul(pe, full_mask)


def is_thing(clipart: str) -> bool:
    """Return True if clipart is a thing (i.e. not a person)."""
    return 'boy' not in clipart and 'girl' not in clipart


def define_monitored_metric(params: Namespace) -> Tuple[str, str]:
    """Return name of metric and mode (min/max) monitored for checkpointing."""
    if params.predict_icrs_turn:
        return 'val_icr_label_BinaryAveragePrecision', 'max'
    if params.predict_icrs_clipart:
        return 'val_icr_clip_label_BinaryAveragePrecision', 'max'
    if not params.dont_make_actions:
        return 'val_action_BinaryAveragePrecision', 'max'
    return 'val_loss', 'min'


def percent(numerator: float, denominator: float) -> float:
    """Return the percentage."""
    return 100 * numerator / denominator


def list_avg(lst):
    """Return the average of a list."""
    return sum(lst)/len(lst)


def filter_checkpoint(state_dict: OrderedDict) -> Dict[str, Tensor]:
    """Remove the 'model' prefix from Lightning's model state dictionary."""
    return {k.replace('model.', ''): v for k, v in state_dict.items()
            if k.startswith('model.')}
