#!/usr/bin/env python3
# -*- coding: utf-8 -*-


config = {
    "paths": {
        "annotation_path": "../data/codraw-icr-v2.tsv",
        "codraw_path": "../data/CoDraw-master/dataset/CoDraw_1_0.json",
        "outputs_path": "./outputs/",
        "token_embeddings_path": "../data/text_embeddings/",
        "scenes_path": "../data/preprocessed/images/raw/"
    },
    "comet": {
        "comet_key": 'aFXiQY8mzPV0ukzBJ6tnPTwjg',
        "comet_project": "codraw-icr-generation",
        "comet_workspace": 'jacob-spiess',
        "ignore_comet": True,
        "comet_tag": ''
    },
    "data": {
        "annotation_path": "../data/codraw-icr-v2.tsv",
        "bart_path": "../data/bart.json",
        "codraw_path": "../data/CoDraw-master/dataset/CoDraw_1_0.json",
        "token_embeddings_path": "../data/text_embeddings/",
        "scenes_path": "../data/preprocessed/images/raw/",
        "context_size": 3,
        "dont_merge_persons": True,
        "langmodel": 'bert-base-uncased',  #['bert-base-uncased', 'roberta-base','distilbert-base-uncased']
        "only_icr_dialogues": True,
        "only_icr_turns": True,
        "score_threshold": 0,
        "reduce_turns_without_actions": True,
        "dont_separate_actions": True
    },
    "generation": {
        "outputs_path": "./outputs/",
        "batch_size": 32,
        "checkpoint": '',
        "lr": 0.0001,
        "pos_weight": 2,
        "scheduler_step": 2,
        "use_weighted_loss": True,
        "use_scheduler": True,
        "weight_decay": 0.0
    },
    "model": {
        "d_model": 256,
        "dropout": 0.1,
        "full_trf_encoder": True,
        "hidden_dim": 1024, 
        "hidden_dim_trf": 1024,
        "dialogue_embedding_size": 512,
        "scene_embedding_size": 512,
        "word_embedding_size": 512,
        "decoder_size": 512,
        "attention_size": 512,
        "topic_classes": 6,
        "num_clip_classes": 5,
        "mood_classes": 7,
        "clipart_classes": 28,
        "nheads": 16,
        "nlayers": 6,
        "lr": 0.0003,
        "use_num_clip_decoder": True,
        "use_clipart_decoder": True,
        "use_topic_decoder": True,
        "use_mood_decoder": True,
        "use_scenes": True, 
        "use_instructions": True,
        "dont_preprocess_scenes": True,
        "unfreeze_resnet": True
    },
    "training": {
        "n_grad_accumulate": 1,
        "clip": 1,
        "device": 'gpu',   #['cpu', 'gpu']
        "gpu": 1,
        "n_epochs": 40,
        "n_reload_data": 1,
        "random_seed": 13
    }
}
