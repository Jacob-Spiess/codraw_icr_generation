#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of several constants, lists and dictionaries use in the experiments.
"""

import json
#import spacy


UNK = "<UNK>" # unknown word
PAD = "<PAD>" # padding
BOS = "<BOS>" # beginning of sentence
EOS = "<EOS>" # end of sentence

#spacy_eng = spacy.load("en_core_web_sm")

NUM_WEIGHTS = [4.7361, 0.0127, 0.0560, 0.0070, 0.1882]
CLIP_WEIGHTS = [0.5277, 0.9288, 0.8468, 0.4276, 0.4429, 0.4843, 0.4627, 0.1154, 0.1154, 0.1154, 0.1154, 0.1154, 0.1127, 0.1127, 
                0.1127, 0.1127, 0.1127, 0.6560, 0.6414, 3.2800, 2.8792, 2.5912, 3.1600, 3.5017, 1.5243, 1.5516, 1.4640, 1.4892]
TOPIC_WEIGHTS = [0.6686, 0.9240, 0.8137, 0.6000, 2.2707, 0.7231]
MOOD_WEIGHTS = [6.6093, 0.0126, 0.0646, 0.2510, 0.0480, 0.0070, 0.0075]

BEFORE_PREFIX = 'before'

AFTER_PREFIX = 'after'

REDUCTION = 'sum'

NA_VALUE = 0

COMET_LOG_PATH = 'comet-logs/'

SPLITS = ('train', 'val', 'test')

ICR_MAP = {'not_icr': 0, 'icr': 1}

# n/a has to be the same as NA_VALUE
FACES = {'n/a': 0, 'angry': 1, 'wide_smile': 2, 'smile': 3,
         'sad': 4, 'scared': 5}

POSES = {'n/a': 0, 'arms_right': 1, 'arms_up': 2, 'kicking': 3, 'running': 4,
         'leg_crossed': 5, 'sit': 6, 'wave': 7}

RESCALING = {0: 1.0, 1: 0.7, 2: 0.49}

EMPTY_SCENES = {'train': 0, 'val': 8, 'test': 9}

with open('../data/clipsizes.json', 'r') as file:
    CLIPSIZES = json.load(file)

N_POSE_CLASSES = len(POSES)

N_FACE_CLASSES = len(FACES)

N_FLIP_CLASSES = 3

N_SIZE_CLASSES = 4

N_PRESENCE_CLASSES = 2

LEN_POSITION = 5

N_CLIPS = 59

RGB_DIM = 255

ACTION_LABELS = ['action', 'action_presence', 'action_move',
                 'action_size', 'action_flip']

N_ACTION_TYPES = len(ACTION_LABELS)

DATA_CONFIG = ['annotation_path', 'codraw_path', 'token_embeddings_path',
               'scenes_path', 'context_size', 'dont_merge_persons',
               'dont_separate_actions', 'langmodel', 'only_icr_dialogues',
               'only_icr_turns', 'reduce_turns_without_actions',
               'score_threshold']

MODEL_CONFIG = ['actions_for_icr', 'd_model', 'dont_make_actions',
                'dont_preprocess_scenes', 'dropout', 'full_trf_encoder',
                'hidden_dim', 'hidden_dim_trf', 'nheads', 'nlayers',
                'no_instruction', 'predict_icrs_turn', 'predict_icrs_clipart',
                'random_baseline', 'unfreeze_resnet', 'use_scene_after',
                'use_scene_before']

EXPERIMENT_CONFIG = ['batch_size', 'checkpoint', 'lr', 'pos_weight',
                     'outputs_path', 'scheduler_step', 'use_weighted_loss',
                     'use_scheduler', 'weight_decay']

TRAINER_CONFIG = ['batch_size', 'clip', 'device', 'gpu', 'n_epochs',
                  'n_grad_accumulate','n_reload_data', 'random_seed']

COMET_CONFIG = ['comet_key', 'comet_project', 'comet_tag',
                'comet_workspace', 'ignore_comet']

LABELS = ['action_presence', 'action_move', 'action_flip', 'action_size',
          'action', 'icr_clip_label', 'icr_label']

OUT_HEADER = ['identifier', 'game_id', 'turn', 'position', 'name', 'clipart',
              'label']
