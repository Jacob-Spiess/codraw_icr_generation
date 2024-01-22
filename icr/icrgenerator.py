#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the experiment with a PytorchLightining model.
"""

from argparse import Namespace
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn, Tensor, optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F
from transformers import BartTokenizer


from icr.aux import mask_pads, split_batch, compute_accuracy_text, compute_accuracy_single_label, compute_accuracy_multi_label
from icr.constants import REDUCTION, SPLITS, BOS, EOS, PAD, NUM_WEIGHTS, CLIP_WEIGHTS, TOPIC_WEIGHTS, MOOD_WEIGHTS
from icr.evaluator import Metrics, Outputs
from icr import constants
from icr.models import (TextEncoder, TextDecoder, EmbeddingCompresser, EmbeddingCompresserLarge, SingleProbDecoder, MultiProbDecoder, 
                        Attention, AttentionTextDecoder, TransformerDecoder, EmbeddingExtender, SceneEncoder, 
                        EmbeddingCompresserCNNScene, EmbeddingCompresserCNNDialogue, BartTransformerDecoder)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ICRModel(pl.LightningModule):
    def __init__(self, vocab, config):
        super(ICRModel, self).__init__()
        
        self.vocab_size = len(vocab)
        max_length = vocab.max_token
        self.vocab = vocab
        self.lr = config["lr"]
        drop = config["dropout"]
        
        self.use_scenes = config["use_scenes"]
        self.use_instructions = config["use_instructions"]
        self.use_num_clip_decoder = config["use_num_clip_decoder"]
        self.use_clipart_decoder = config["use_clipart_decoder"]
        self.use_topic_decoder = config["use_topic_decoder"]
        self.use_mood_decoder = config["use_mood_decoder"]

        self.loss_cross = CrossEntropyLoss(reduction=REDUCTION)
        self.loss_cross_n = CrossEntropyLoss(weight=torch.tensor(NUM_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_c = BCEWithLogitsLoss(pos_weight=torch.tensor(CLIP_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_t = BCEWithLogitsLoss(pos_weight=torch.tensor(TOPIC_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_m = BCEWithLogitsLoss(pos_weight=torch.tensor(MOOD_WEIGHTS), reduction=REDUCTION)
        
        num_clip_decoder_size = (
            config["decoder_size"] if self.use_instructions and not self.use_scenes
            else config["decoder_size"] * 2 if not self.use_instructions and self.use_scenes
            else config["decoder_size"] * 3 if self.use_instructions and self.use_scenes
            else print("No input available"))
        clipart_decoder_size = num_clip_decoder_size + config["num_clip_classes"] if self.use_num_clip_decoder else num_clip_decoder_size
        topic_decoder_size = clipart_decoder_size + config["clipart_classes"] if self.use_clipart_decoder else clipart_decoder_size
        mood_decoder_size = topic_decoder_size + config["topic_classes"] if self.use_topic_decoder else topic_decoder_size
        context_size = mood_decoder_size + config["mood_classes"] if self.use_mood_decoder else mood_decoder_size

        self.text_encoder = TextEncoder(self.vocab_size, config["hidden_dim"], config["decoder_size"], drop, self.vocab.stoi[PAD])
        self.pe = PositionalEncoding1D(config["decoder_size"])
        self.scene_encoder = SceneEncoder(d_model=config["decoder_size"], dropout=drop, unfreeze=config["unfreeze_resnet"], 
                                          dont_preprocess_scenes=config["dont_preprocess_scenes"])
        self.dialogue_encoder = EmbeddingCompresser(config["decoder_size"])
        self.num_clip_decoder = (SingleProbDecoder(num_clip_decoder_size, config["hidden_dim"], config["num_clip_classes"],drop)
                                if self.use_num_clip_decoder
                                else None)
        self.n_mapping = EmbeddingExtender(config["num_clip_classes"], config["decoder_size"])
        self.clipart_decoder = (MultiProbDecoder(clipart_decoder_size, config["hidden_dim"], config["clipart_classes"], drop)
                                if self.use_clipart_decoder
                                else None)
        self.c_mapping = EmbeddingExtender(config["clipart_classes"], config["decoder_size"])
        self.topic_decoder = (MultiProbDecoder(topic_decoder_size, config["hidden_dim"], config["topic_classes"], drop)
                                if self.use_topic_decoder
                                else None)
        self.t_mapping = EmbeddingExtender(config["topic_classes"], config["decoder_size"])
        self.mood_decoder = (MultiProbDecoder(mood_decoder_size, config["hidden_dim"], config["mood_classes"], drop)
                                if self.use_mood_decoder
                                else None)
        self.m_mapping = EmbeddingExtender(config["mood_classes"], config["decoder_size"])

        self.decoder = TransformerDecoder(config["decoder_size"], config["hidden_dim_trf"], self.vocab_size, config["nheads"], 
                                          config["nlayers"], drop)
        self.scene_compresser = EmbeddingCompresserCNNScene(config["decoder_size"], config["hidden_dim"], config["decoder_size"])
        self.dialogue_compresser = EmbeddingCompresserCNNDialogue(config["hidden_dim"], config["decoder_size"])

        self.save_hyperparameters()
        self.to(device)
        

    def forward(self, input_sequence, dialogue, scene_before, scene_after):

        features1D = []
        features2D = []
        if self.use_instructions:
            dialogue_features = self.dialogue_encoder(dialogue)
            positions = mask_pads(dialogue_features, self.pe(dialogue_features))
            dialogue_features = torch.cat([dialogue_features + positions],dim=1)
            features2D.append(dialogue_features)
            features1D.append(self.dialogue_compresser(dialogue))
        if self.use_scenes:
            scene_before_features = self.scene_encoder(scene_before)
            scene_after_features = self.scene_encoder(scene_after)
            scenes = torch.cat((scene_before_features, scene_after_features), dim=1)
            positions = self.pe(scenes)
            scene_features = torch.cat([scenes + positions], dim=1)
            features2D.append(scene_features)
            features1D.append(self.scene_compresser(scene_before_features)) 
            features1D.append(self.scene_compresser(scene_after_features))

        features = torch.cat(features1D, dim=1).to(device)
        padding_mask = input_sequence == self.vocab.stoi[PAD]
        embeds = self.text_encoder(input_sequence).to(device)

        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features)
            features = torch.cat((features, n), dim=1)
            n_features = self.n_mapping(n).unsqueeze(1)
            features2D.append(n_features)
        else:
            n = None
            
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features)
            features = torch.cat((features, c), dim=1)
            c_features = self.c_mapping(c).unsqueeze(1)
            features2D.append(c_features)
        else:
            c = None
        
        if self.use_topic_decoder:
            t = self.topic_decoder(features)
            features = torch.cat((features, t), dim=1)
            t_features = self.t_mapping(t).unsqueeze(1)
            features2D.append(t_features)
        else:
            t = None
        
        if self.use_mood_decoder:
            m = self.mood_decoder(features)
            features = torch.cat((features, m), dim=1)
            m_features = self.m_mapping(m).unsqueeze(1)
            features2D.append(m_features)
        else:
            m = None

        features = torch.cat(features2D, dim=1).to(device)
        outputs = self.decoder(embeds, features, padding_mask)
        
        return outputs, n, c, t, m
    
    def training_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)
        
        drawer_sequence = drawer_sequence.to(device)
        targets = targets.to(device)
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)
        mood = mood.to(device)
        num_clip = num_clip.to(device)
        topic = topic.to(device)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        self.log_metric("train", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        return loss + loss_n + loss_c + loss_t + loss_m 

    def validation_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)
        
        drawer_sequence = drawer_sequence.to(device)
        targets = targets.to(device)
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)
        mood = mood.to(device)
        num_clip = num_clip.to(device)
        topic = topic.to(device)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        acc, acc_n, acc_c, acc_t, acc_m = self.calculate_acc(outputs, targets, n, num_clip, c, clip, t, topic, 
                                                             m, mood,self.vocab.stoi[PAD])

        self.log_metric("val", "acc", acc, acc_n, acc_c, acc_t, acc_m)
        self.log_metric("val", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 
    
    def test_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)
        
        drawer_sequence = drawer_sequence.to(device)
        targets = targets.to(device)
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)
        mood = mood.to(device)
        num_clip = num_clip.to(device)
        topic = topic.to(device)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        acc, acc_n, acc_c, acc_t, acc_m = self.calculate_acc(outputs, targets, n, num_clip, c, clip, t, topic, 
                                                             m, mood,self.vocab.stoi[PAD])

        self.log_metric("test", "acc", acc, acc_n, acc_c, acc_t, acc_m)
        self.log_metric("test", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer 
    
    def calculate_loss(self, outputs, targets, n, num_clip, c, clip, t, topic, m, mood):
        targets=targets.reshape(-1)
        outputs=outputs.view(-1, outputs.size(-1))
        
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in self.parameters())

        loss = self.loss_cross(outputs, targets) + l1_lambda * l1_norm
        loss_n = self.loss_cross_n(n, num_clip.argmax(dim=1)) if self.use_num_clip_decoder and n is not None else 0.0
        loss_c = self.loss_bce_c(c, clip.float()) if self.use_clipart_decoder and c is not None else 0.0
        loss_t = self.loss_bce_t(t, topic.float()) if self.use_topic_decoder and t is not None else 0.0
        loss_m = self.loss_bce_m(m, mood.float()) if self.use_mood_decoder and m is not None else 0.0
        
        return loss, loss_n, loss_c, loss_t, loss_m
        
    def calculate_acc(self, outputs, targets, n, num_clip, c, clip, t, topic, m, mood, pad):        
        acc = compute_accuracy_text(outputs, targets, pad)
        acc_n = compute_accuracy_single_label(n, num_clip.argmax(dim=1)) if self.use_num_clip_decoder and n is not None else 0.0
        acc_c = compute_accuracy_multi_label(c, clip.float()) if self.use_clipart_decoder and c is not None else 0.0
        acc_t = compute_accuracy_multi_label(t, topic.float()) if self.use_topic_decoder and t is not None else 0.0
        acc_m = compute_accuracy_multi_label(m, mood.float()) if self.use_mood_decoder and m is not None else 0.0
        
        return acc, acc_n, acc_c, acc_t, acc_m
    
    def log_metric(self, step, metric, total, n, c, t, m):
        self.log(f'{step}_{metric}', total)
        self.log(f'{step}_{metric}_n', n)
        self.log(f'{step}_{metric}_c', c)
        self.log(f'{step}_{metric}_t', t)
        self.log(f'{step}_{metric}_m', m)
        return

    def top_k_top_p_filtering(self, logits, top_k, top_p, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
            
        return logits

    def reply(self, dialogue, scene_before, scene_after, use_sampling: bool, top_k: int, top_p: float, temperature: float, max_len = 39):
        self.text_encoder.to(device)
        self.num_clip_decoder.to(device)
        self.clipart_decoder.to(device)
        self.topic_decoder.to(device)
        self.mood_decoder.to(device)
        self.n_mapping.to(device), self.c_mapping.to(device), self.t_mapping.to(device), self.m_mapping.to(device)
        self.decoder.to(device)
        dialogue.to(device)
        scene_before.to(device)
        scene_after.to(device)
        
        word = torch.tensor(self.vocab.stoi[BOS]).view(1,-1).to(device)
        embeds = self.text_encoder(word)
        
        features1D = []
        features2D = []
        if self.use_instructions:
            dialogue_features = self.dialogue_encoder(dialogue)
            positions = mask_pads(dialogue_features, self.pe(dialogue_features))
            dialogue_features = torch.cat([dialogue_features + positions],dim=1)
            features2D.append(dialogue_features)
            features1D.append(self.dialogue_compresser(dialogue))
        if self.use_scenes:
            scene_before_features = self.scene_encoder(scene_before)
            scene_after_features = self.scene_encoder(scene_after)
            scenes = torch.cat((scene_before_features, scene_after_features), dim=1)
            positions = self.pe(scenes)
            scene_features = torch.cat([scenes + positions], dim=1)
            features2D.append(scene_features)
            features1D.append(self.scene_compresser(scene_before_features)) 
            features1D.append(self.scene_compresser(scene_after_features))

        features = torch.cat(features1D, dim=1).to(device)
              
        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features).to(device)
            features = torch.cat((features, n), dim=1)
            n_features = self.n_mapping(n).unsqueeze(1)
            features2D.append(n_features)
        else:
            n = None
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features).to(device)
            features = torch.cat((features, c), dim=1)
            c_features = self.c_mapping(c).unsqueeze(1)
            features2D.append(c_features)
        else:
            c = None
        if self.use_topic_decoder:
            t = self.topic_decoder(features).to(device)
            features = torch.cat((features, t), dim=1)
            t_features = self.t_mapping(t).unsqueeze(1)
            features2D.append(t_features)
        else:
            t = None
        if self.use_mood_decoder:
            m = self.mood_decoder(features).to(device)
            features = torch.cat((features, m), dim=1)
            m_features = self.m_mapping(m).unsqueeze(1)
            features2D.append(m_features)
        else:
            m = None
            
        features = torch.cat(features2D, dim=1).to(device)
        reply = []
        outputs = []
        outputs_tensor = torch.zeros(1,max_len, self.vocab_size).to(device)

        for i in range(max_len):
            h = self.decoder.transformer(embeds, features)
            output = self.decoder.fc(self.decoder.dropout(h)).view(embeds.size(0),-1).to(device)
            outputs.append(output.view(-1, output.size(-1)))
            
            if use_sampling:
                output = output / temperature
                filtered_logits = self.top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)
                prediction_index = torch.multinomial(probabilities, 1).squeeze(1)
            else:
                prediction_index = output.argmax(dim=1)
            
            reply.append(self.vocab.itos[prediction_index.item()])

            if self.vocab.itos[prediction_index.item()] == EOS:
                l = max_len - i -1
                outputs = torch.stack(outputs, dim = 1).to(device)
                outputs_tensor = torch.cat((outputs.clone().detach(), torch.zeros(1, l, self.vocab_size).to(device)), dim=1)
                break
            else:
                embeds = self.text_encoder(prediction_index.unsqueeze(0))

        return reply, outputs_tensor, n, c, t, m

    
class ICRModelBart(pl.LightningModule):
    def __init__(self, vocab, config):
        super(ICRModelBart, self).__init__()
        
        self.vocab_size = len(vocab)
        max_length = vocab.max_token
        self.vocab = vocab
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.lr = config["lr"]
        drop = config["dropout"]
        
        self.use_scenes = config["use_scenes"]
        self.use_instructions = config["use_instructions"]
        self.use_num_clip_decoder = config["use_num_clip_decoder"]
        self.use_clipart_decoder = config["use_clipart_decoder"]
        self.use_topic_decoder = config["use_topic_decoder"]
        self.use_mood_decoder = config["use_mood_decoder"]

        self.loss_cross = CrossEntropyLoss(reduction=REDUCTION)
        self.loss_cross_n = CrossEntropyLoss(weight=torch.tensor(NUM_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_c = BCEWithLogitsLoss(pos_weight=torch.tensor(CLIP_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_t = BCEWithLogitsLoss(pos_weight=torch.tensor(TOPIC_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_m = BCEWithLogitsLoss(pos_weight=torch.tensor(MOOD_WEIGHTS), reduction=REDUCTION)
        
        num_clip_decoder_size = (
            config["decoder_size"] if self.use_instructions and not self.use_scenes
            else config["decoder_size"] * 2 if not self.use_instructions and self.use_scenes
            else config["decoder_size"] * 3 if self.use_instructions and self.use_scenes
            else print("No model input available"))
        clipart_decoder_size = num_clip_decoder_size + config["num_clip_classes"] if self.use_num_clip_decoder else num_clip_decoder_size
        topic_decoder_size = clipart_decoder_size + config["clipart_classes"] if self.use_clipart_decoder else clipart_decoder_size
        mood_decoder_size = topic_decoder_size + config["topic_classes"] if self.use_topic_decoder else topic_decoder_size
        context_size = mood_decoder_size + config["mood_classes"] if self.use_mood_decoder else mood_decoder_size

        self.text_encoder = TextEncoder(self.vocab_size, config["hidden_dim"], config["decoder_size"], drop, self.tokenizer.pad_token_id)
        self.pe = PositionalEncoding1D(config["decoder_size"])
        self.scene_encoder = SceneEncoder(d_model=config["decoder_size"], dropout=drop, unfreeze=config["unfreeze_resnet"], 
                                          dont_preprocess_scenes=config["dont_preprocess_scenes"])
        self.dialogue_encoder = EmbeddingCompresser(config["decoder_size"])
        self.num_clip_decoder = (SingleProbDecoder(num_clip_decoder_size, config["hidden_dim"], config["num_clip_classes"],drop)
                                if self.use_num_clip_decoder
                                else None)
        self.n_mapping = EmbeddingExtender(config["num_clip_classes"], config["decoder_size"])
        self.clipart_decoder = (MultiProbDecoder(clipart_decoder_size, config["hidden_dim"], config["clipart_classes"], drop)
                                if self.use_clipart_decoder
                                else None)
        self.c_mapping = EmbeddingExtender(config["clipart_classes"], config["decoder_size"])
        self.topic_decoder = (MultiProbDecoder(topic_decoder_size, config["hidden_dim"], config["topic_classes"], drop)
                                if self.use_topic_decoder
                                else None)
        self.t_mapping = EmbeddingExtender(config["topic_classes"], config["decoder_size"])
        self.mood_decoder = (MultiProbDecoder(mood_decoder_size, config["hidden_dim"], config["mood_classes"], drop)
                                if self.use_mood_decoder
                                else None)
        self.m_mapping = EmbeddingExtender(config["mood_classes"], config["decoder_size"])

        self.decoder = BartTransformerDecoder(self.vocab_size, config["decoder_size"], config["hidden_dim_trf"], 
                                              config["nheads"], config["nlayers"], drop)
        self.scene_compresser = EmbeddingCompresserCNNScene(config["decoder_size"], config["hidden_dim"], config["decoder_size"])
        self.dialogue_compresser = EmbeddingCompresserCNNDialogue(config["hidden_dim"], config["decoder_size"]) 

        self.save_hyperparameters()
        self.to(device)
        

    def forward(self, input_sequence, dialogue, scene_before, scene_after):

        features1D = []
        features2D = []
        if self.use_instructions:
            dialogue_features = self.dialogue_encoder(dialogue)
            positions = mask_pads(dialogue_features, self.pe(dialogue_features))
            dialogue_features = torch.cat([dialogue_features + positions],dim=1)
            features2D.append(dialogue_features)
            features1D.append(self.dialogue_compresser(dialogue))
        if self.use_scenes:
            scene_before_features = self.scene_encoder(scene_before)
            scene_after_features = self.scene_encoder(scene_after)
            scenes = torch.cat((scene_before_features, scene_after_features), dim=1)
            positions = self.pe(scenes)
            scene_features = torch.cat([scenes + positions], dim=1)
            features2D.append(scene_features)
            features1D.append(self.scene_compresser(scene_before_features))
            features1D.append(self.scene_compresser(scene_after_features)) 

        features = torch.cat(features1D, dim=1).to(device)
        padding_mask = input_sequence == self.tokenizer.pad_token_id
        embeds = self.text_encoder(input_sequence).to(device)

        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features)
            features = torch.cat((features, n), dim=1)
            n_features = self.n_mapping(n).unsqueeze(1)
            features2D.append(n_features)
        else:
            n = None
            
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features)
            features = torch.cat((features, c), dim=1)
            c_features = self.c_mapping(c).unsqueeze(1)
            features2D.append(c_features)
        else:
            c = None
        
        if self.use_topic_decoder:
            t = self.topic_decoder(features)
            features = torch.cat((features, t), dim=1)
            t_features = self.t_mapping(t).unsqueeze(1)
            features2D.append(t_features)
        else:
            t = None
        
        if self.use_mood_decoder:
            m = self.mood_decoder(features)
            features = torch.cat((features, m), dim=1)
            m_features = self.m_mapping(m).unsqueeze(1)
            features2D.append(m_features)
        else:
            m = None

        features = torch.cat(features2D, dim=1).to(device)
        outputs = self.decoder(embeds, features)
        
        return outputs, n, c, t, m
    
    def training_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch, bart=True)
        
        drawer_sequence = drawer_sequence.to(device)
        targets = targets.to(device)
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)
        mood = mood.to(device)
        num_clip = num_clip.to(device)
        topic = topic.to(device)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        self.log_metric("train", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        return loss + loss_n + loss_c + loss_t + loss_m 

    def validation_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)
        
        drawer_sequence = drawer_sequence.to(device)
        targets = targets.to(device)
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)
        mood = mood.to(device)
        num_clip = num_clip.to(device)
        topic = topic.to(device)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        acc, acc_n, acc_c, acc_t, acc_m = self.calculate_acc(outputs, targets, n, num_clip, c, clip, t, topic, 
                                                             m, mood,self.vocab.stoi[PAD])

        self.log_metric("val", "acc", acc, acc_n, acc_c, acc_t, acc_m)
        self.log_metric("val", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 
    
    def test_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)
        
        drawer_sequence = drawer_sequence.to(device)
        targets = targets.to(device)
        dialogue = dialogue.to(device)
        scene_before = scene_before.to(device)
        scene_after = scene_after.to(device)
        clip = clip.to(device)
        mood = mood.to(device)
        num_clip = num_clip.to(device)
        topic = topic.to(device)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        acc, acc_n, acc_c, acc_t, acc_m = self.calculate_acc(outputs, targets, n, num_clip, c, clip, t, topic, 
                                                             m, mood,self.vocab.stoi[PAD])

        self.log_metric("test", "acc", acc, acc_n, acc_c, acc_t, acc_m)
        self.log_metric("test", "loss", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
    
    def calculate_loss(self, outputs, targets, n, num_clip, c, clip, t, topic, m, mood):
        targets=targets.reshape(-1)
        outputs=outputs.view(-1, outputs.size(-1))
        
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in self.parameters())

        loss = self.loss_cross(outputs, targets) + l1_lambda * l1_norm
        loss_n = self.loss_cross_n(n, num_clip.argmax(dim=1)) if self.use_num_clip_decoder and n is not None else 0.0
        loss_c = self.loss_bce_c(c, clip.float()) if self.use_clipart_decoder and c is not None else 0.0
        loss_t = self.loss_bce_t(t, topic.float()) if self.use_topic_decoder and t is not None else 0.0
        loss_m = self.loss_bce_m(m, mood.float()) if self.use_mood_decoder and m is not None else 0.0
        
        return loss, loss_n, loss_c, loss_t, loss_m
        
    def calculate_acc(self, outputs, targets, n, num_clip, c, clip, t, topic, m, mood, pad):        
        acc = compute_accuracy_text(outputs, targets, pad)
        acc_n = compute_accuracy_single_label(n, num_clip.argmax(dim=1)) if self.use_num_clip_decoder and n is not None else 0.0
        acc_c = compute_accuracy_multi_label(c, clip.float()) if self.use_clipart_decoder and c is not None else 0.0
        acc_t = compute_accuracy_multi_label(t, topic.float()) if self.use_topic_decoder and t is not None else 0.0
        acc_m = compute_accuracy_multi_label(m, mood.float()) if self.use_mood_decoder and m is not None else 0.0
        
        return acc, acc_n, acc_c, acc_t, acc_m
    
    def log_metric(self, step, metric, total, n, c, t, m):
        self.log(f'{step}_{metric}', total)
        self.log(f'{step}_{metric}_n', n)
        self.log(f'{step}_{metric}_c', c)
        self.log(f'{step}_{metric}_t', t)
        self.log(f'{step}_{metric}_m', m)
        return

    def top_k_top_p_filtering(self, logits, top_k, top_p, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
            
        return logits

    def reply(self, dialogue, scene_before, scene_after, use_sampling: bool, top_k: int, top_p: float, temperature: float, max_len = 39):
        self.text_encoder.to(device)
        self.num_clip_decoder.to(device)
        self.clipart_decoder.to(device)
        self.topic_decoder.to(device)
        self.mood_decoder.to(device)
        self.n_mapping.to(device), self.c_mapping.to(device), self.t_mapping.to(device), self.m_mapping.to(device)
        self.decoder.to(device)
        dialogue.to(device)
        scene_before.to(device)
        scene_after.to(device)
        
        word = torch.tensor(self.tokenizer.bos_token_id).view(1,-1).to(device)
        embeds = self.text_encoder(word)
        
        features1D = []
        features2D = []
        if self.use_instructions:
            dialogue_features = self.dialogue_encoder(dialogue)
            positions = mask_pads(dialogue_features, self.pe(dialogue_features))
            dialogue_features = torch.cat([dialogue_features + positions],dim=1)
            features2D.append(dialogue_features)
            features1D.append(self.dialogue_compresser(dialogue))
        if self.use_scenes:
            scene_before_features = self.scene_encoder(scene_before)
            scene_after_features = self.scene_encoder(scene_after)
            scenes = torch.cat((scene_before_features, scene_after_features), dim=1)
            positions = self.pe(scenes)
            scene_features = torch.cat([scenes + positions], dim=1)
            features2D.append(scene_features)
            features1D.append(self.scene_compresser(scene_before_features)) 
            features1D.append(self.scene_compresser(scene_after_features)) 

        features = torch.cat(features1D, dim=1).to(device)
              
        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features).to(device)
            features = torch.cat((features, n), dim=1)
            n_features = self.n_mapping(n).unsqueeze(1)
            features2D.append(n_features)
        else:
            n = None
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features).to(device)
            features = torch.cat((features, c), dim=1)
            c_features = self.c_mapping(c).unsqueeze(1)
            features2D.append(c_features)
        else:
            c = None
        if self.use_topic_decoder:
            t = self.topic_decoder(features).to(device)
            features = torch.cat((features, t), dim=1)
            t_features = self.t_mapping(t).unsqueeze(1)
            features2D.append(t_features)
        else:
            t = None
        if self.use_mood_decoder:
            m = self.mood_decoder(features).to(device)
            features = torch.cat((features, m), dim=1)
            m_features = self.m_mapping(m).unsqueeze(1)
            features2D.append(m_features)
        else:
            m = None
            
        features = torch.cat(features2D, dim=1).to(device)
        reply = []
        outputs = []
        outputs_tensor = torch.zeros(1,max_len, self.vocab_size).to(device)

        for i in range(max_len):
            h = self.decoder.bart(encoder_hidden_states=memory, inputs_embeds= target)
            outputs = self.decoder.fc(self.dropout(h.hidden_states[1])).view(embeds.size(0),-1).to(device)
            outputs.append(output.view(-1, output.size(-1)))
            
            if use_sampling:
                output = output / temperature
                filtered_logits = self.top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)
                prediction_index = torch.multinomial(probabilities, 1).squeeze(1)
            else:
                prediction_index = output.argmax(dim=1)
            
            reply.append(prediction_index.item())

            if prediction_index.item() == self.tokenizer.eos_token_id:
                l = max_len - i - 1
                outputs = torch.stack(outputs, dim = 1).to(device)
                #outputs_tensor = torch.cat((torch.tensor(outputs), torch.zeros(1, l, self.vocab_size)), dim=1)
                outputs_tensor = torch.cat((outputs.clone().detach(), torch.zeros(1, l, self.vocab_size).to(device)), dim=1)
                break
            else:
                embeds = self.text_encoder(prediction_index.unsqueeze(0))
        reply = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(reply))
        return reply, outputs_tensor, n, c, t, m
    
    
    
class ICRModel1(pl.LightningModule):
    def __init__(self, vocab, config):
        super(ICRModel1, self).__init__()
        
        self.vocab_size = len(vocab)
        max_length = vocab.max_token
        self.vocab = vocab
        self.lr = config["lr"]
        drop = config["dropout"]
        
        self.use_num_clip_decoder = config["use_num_clip_decoder"]
        self.use_clipart_decoder = config["use_clipart_decoder"]
        self.use_topic_decoder = config["use_topic_decoder"]
        self.use_mood_decoder = config["use_mood_decoder"]

        self.loss_cross = CrossEntropyLoss(reduction=REDUCTION)
        self.loss_cross_n = CrossEntropyLoss(weight=torch.tensor(NUM_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_c = BCEWithLogitsLoss(pos_weight=torch.tensor(CLIP_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_t = BCEWithLogitsLoss(pos_weight=torch.tensor(TOPIC_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_m = BCEWithLogitsLoss(pos_weight=torch.tensor(MOOD_WEIGHTS), reduction=REDUCTION)
        
        num_clip_decoder_size = config["dialogue_embedding_size"]+2*config["scene_embedding_size"]
        clip_decoder_size = num_clip_decoder_size + config["num_clip_classes"] if self.use_num_clip_decoder else num_clip_decoder_size
        topic_decoder_size = clip_decoder_size + config["clipart_classes"] if self.use_clipart_decoder else clip_decoder_size
        mood_decoder_size = topic_decoder_size + config["topic_classes"] if self.use_topic_decoder else topic_decoder_size
        context_size = mood_decoder_size + config["mood_classes"] if self.use_mood_decoder else mood_decoder_size

        self.text_encoder = TextEncoder(self.vocab_size, config["hidden_dim"], config["word_embedding_size"], drop)
        self.dialogue_encoder = EmbeddingCompresserCNNDialogue(config["hidden_dim"], config["dialogue_embedding_size"]) 
        self.scene_encoder = SceneEncoder(d_model=config["decoder_size"], dropout=drop, unfreeze=config["unfreeze_resnet"], 
                                          dont_preprocess_scenes=config["dont_preprocess_scenes"])
        self.scene_compresser = EmbeddingCompresserCNNScene(config["decoder_size"], config["hidden_dim"], config["scene_embedding_size"])
        self.num_clip_decoder = (SingleProbDecoder(num_clip_decoder_size, config["hidden_dim"],config["num_clip_classes"],drop)
                                if self.use_num_clip_decoder
                                else None)
        self.clipart_decoder = (MultiProbDecoder(clip_decoder_size, config["hidden_dim"], config["clipart_classes"], drop)
                                if self.use_clipart_decoder
                                else None)
        self.topic_decoder = (MultiProbDecoder(topic_decoder_size, config["hidden_dim"], config["topic_classes"], drop)
                                if self.use_topic_decoder
                                else None)
        self.mood_decoder = (MultiProbDecoder(mood_decoder_size, config["hidden_dim"], config["mood_classes"], drop)
                                if self.use_mood_decoder
                                else None)

        self.decoder = TextDecoder(config["word_embedding_size"], context_size, self.vocab_size, config["nlayers"], drop)
        
        self.save_hyperparameters()


    def forward(self, input_sequence, dialogue, scene_before, scene_after):
        
        f_dialogue = self.dialogue_encoder(dialogue)
        f_scene_before = self.scene_encoder(scene_before)
        f_scene_after = self.scene_encoder(scene_after)
        f_scene_before = self.scene_compresser(f_scene_before)
        f_scene_after = self.scene_compresser(f_scene_after)
        features = torch.cat((f_dialogue, f_scene_before, f_scene_after), dim=1)
        embeds = self.text_encoder(input_sequence)

        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features)
            features = torch.cat((features, n), dim=1)
        else:
            n = None
            
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features)
            features = torch.cat((features, c), dim=1)
        else:
            c = None
        
        if self.use_topic_decoder:
            t = self.topic_decoder(features)
            features = torch.cat((features, t), dim=1)
        else:
            t = None
        
        if self.use_mood_decoder:
            m = self.mood_decoder(features)
            features = torch.cat((features, m), dim=1)
        else:
            m = None
        
        outputs = self.decoder(embeds, features)
        
        return outputs, n, c, t, m
    
    def training_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)
        
        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        self.log_loss("train", loss, loss_n, loss_c, loss_t, loss_m)
        return loss + loss_n + loss_c + loss_t + loss_m 

    def validation_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)

        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        
        acc = compute_accuracy_text(outputs, targets, self.vocab.stoi[PAD])
        self.log('val_acc', acc, prog_bar=True)
        self.log_loss("val", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 
    
    def test_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        acc = compute_accuracy_text(outputs, targets, self.vocab.stoi[PAD])
        self.log('test_acc', acc, prog_bar=True)
        self.log_loss("test", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
    
    def calculate_loss(self, outputs, targets, n, num_clip, c, clip, t, topic, m, mood):
        targets=targets.reshape(-1)
        outputs=outputs.view(-1, outputs.size(-1))
        
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in self.parameters())

        loss = self.loss_cross(outputs, targets) + l1_lambda * l1_norm
        loss_n = self.loss_cross_n(n, num_clip.argmax(dim=1)) if self.use_num_clip_decoder and n is not None else 0.0
        loss_c = self.loss_bce_c(c, clip.float()) if self.use_clipart_decoder and c is not None else 0.0
        loss_t = self.loss_bce_t(t, topic.float()) if self.use_topic_decoder and t is not None else 0.0
        loss_m = self.loss_bce_m(m, mood.float()) if self.use_mood_decoder and m is not None else 0.0
        
        return loss, loss_n, loss_c, loss_t, loss_m
        
    def log_loss(self, step, loss, loss_n, loss_c, loss_t, loss_m):
        self.log(f'{step}_loss', loss)
        self.log(f'{step}_loss_n', loss_n)
        self.log(f'{step}_loss_c', loss_c)
        self.log(f'{step}_loss_t', loss_t)
        self.log(f'{step}_loss_m', loss_m)
        return

    def reply(self, dialogue, scene_before, scene_after, use_sampling: bool, top_k: int, top_p: float, temperature: float, max_len = 39):
        word = torch.tensor(self.vocab.stoi[BOS]).view(1,-1).to(device)
        embeds = self.text_encoder(word)
        
        f_dialogue = self.dialogue_encoder(dialogue)
        f_scene_before = self.scene_encoder(scene_before)
        f_scene_after = self.scene_encoder(scene_after)
        f_scene_before = self.scene_compresser(f_scene_before)
        f_scene_after = self.scene_compresser(f_scene_after)
        features = torch.cat((f_dialogue, f_scene_before, f_scene_after), dim=1)

        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features)
            features = torch.cat((features, n), dim=1)
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features)
            features = torch.cat((features, c), dim=1)
        if self.use_topic_decoder:
            t = self.topic_decoder(features)
            features = torch.cat((features, t), dim=1)
        if self.use_mood_decoder:
            m = self.mood_decoder(features)
            features = torch.cat((features, m), dim=1)
        
        h0 = features.repeat(self.decoder.num_layers, 1, 1) 
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)

        reply = []
        outputs = []
        outputs_tensor = torch.zeros(1,max_len, self.vocab_size).to(device)

        for i in range(max_len):
            x, hidden = self.decoder.lstm(embeds, hidden)
            output = self.decoder.fc(x).view(embeds.size(0),-1)
            outputs.append(output.view(-1, output.size(-1)))
            
            prediction_index = output.argmax(dim=1)
            reply.append(self.vocab.itos[prediction_index.item()])

            if self.vocab.itos[prediction_index.item()] == EOS:
                l = max_len - i -1
                outputs = torch.stack(outputs, dim = 1).to(device)
                outputs_tensor = torch.cat((outputs.clone().detach(), torch.zeros(1, l, self.vocab_size).to(device)), dim=1)
                break
            else:
                embeds = self.text_encoder(prediction_index.unsqueeze(0))

        return reply, outputs_tensor, n, c, t, m
    


class ICRModel2(pl.LightningModule):
    def __init__(self, vocab, config):
        super(ICRModel2, self).__init__()
        
        self.vocab_size = len(vocab)
        max_length = vocab.max_token
        self.vocab = vocab
        self.lr = config["lr"]
        drop = config["dropout"]
        
        self.use_num_clip_decoder = config["use_num_clip_decoder"]
        self.use_clipart_decoder = config["use_clipart_decoder"]
        self.use_topic_decoder = config["use_topic_decoder"]
        self.use_mood_decoder = config["use_mood_decoder"]

        self.loss_cross = CrossEntropyLoss(reduction=REDUCTION)
        self.loss_cross_n = CrossEntropyLoss(weight=torch.tensor(NUM_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_c = BCEWithLogitsLoss(pos_weight=torch.tensor(CLIP_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_t = BCEWithLogitsLoss(pos_weight=torch.tensor(TOPIC_WEIGHTS), reduction=REDUCTION)
        self.loss_bce_m = BCEWithLogitsLoss(pos_weight=torch.tensor(MOOD_WEIGHTS), reduction=REDUCTION)
        
        num_clip_decoder_size = config["dialogue_embedding_size"]+2*config["scene_embedding_size"]
        clip_decoder_size = num_clip_decoder_size + config["num_clip_classes"] if self.use_num_clip_decoder else num_clip_decoder_size
        topic_decoder_size = clip_decoder_size + config["clipart_classes"] if self.use_clipart_decoder else clip_decoder_size
        mood_decoder_size = topic_decoder_size + config["topic_classes"] if self.use_topic_decoder else topic_decoder_size
        context_size = mood_decoder_size + config["mood_classes"] if self.use_mood_decoder else mood_decoder_size

        self.text_encoder = TextEncoder(self.vocab_size, config["hidden_dim"], config["word_embedding_size"], drop)
        self.dialogue_encoder = EmbeddingCompresserLarge(320*768, config["dialogue_embedding_size"])
        self.scene_encoder = SceneEncoder(d_model=config["decoder_size"], dropout=drop, unfreeze=config["unfreeze_resnet"], 
                                          dont_preprocess_scenes=config["dont_preprocess_scenes"])
        self.scene_compresser = EmbeddingCompresserCNNScene(config["scene_embedding_size"])
        self.num_clip_decoder = (SingleProbDecoder(num_clip_decoder_size, config["hidden_dim"],config["num_clip_classes"],drop)
                                if self.use_num_clip_decoder
                                else None)
        self.clipart_decoder = (MultiProbDecoder(clip_decoder_size, config["hidden_dim"], config["clipart_classes"], drop)
                                if self.use_clipart_decoder
                                else None)
        self.topic_decoder = (MultiProbDecoder(topic_decoder_size, config["hidden_dim"], config["topic_classes"], drop)
                                if self.use_topic_decoder
                                else None)
        self.mood_decoder = (MultiProbDecoder(mood_decoder_size, config["hidden_dim"], config["mood_classes"], drop)
                                if self.use_mood_decoder
                                else None)
        
        self.decoder = AttentionTextDecoder(config["word_embedding_size"], context_size, config["decoder_size"], config["attention_size"], 
                                            self.vocab_size, vocab, drop)
        
        self.save_hyperparameters()


    def forward(self, input_sequence, dialogue, scene_before, scene_after):
        
        f_dialogue = self.dialogue_encoder(dialogue)
        f_scene_before = self.scene_compresser(self.scene_encoder(scene_before))
        f_scene_after = self.scene_compresser(self.scene_encoder(scene_after))
        features = torch.cat((f_dialogue, f_scene_before, f_scene_after), dim=1)  
        embeds = self.text_encoder(input_sequence)

        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features)
            features = torch.cat((features, n), dim=1)
        else:
            n = None
            
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features)
            features = torch.cat((features, c), dim=1)
        else:
            c = None
        
        if self.use_topic_decoder:
            t = self.topic_decoder(features)
            features = torch.cat((features, t), dim=1)
        else:
            t = None
        
        if self.use_mood_decoder:
            m = self.mood_decoder(features)
            features = torch.cat((features, m), dim=1)
        else:
            m = None
        
        outputs = self.decoder(embeds, features)
        
        return outputs[0], n, c, t, m
    
    def training_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)
        
        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        outputs = outputs.to('cuda:0')
        targets = targets.to('cuda:0')
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        self.log_loss("train", loss, loss_n, loss_c, loss_t, loss_m)
        return loss + loss_n + loss_c + loss_t + loss_m 

    def validation_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)

        outputs = outputs.to('cuda:0')
        targets = targets.to('cuda:0')
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)
        
        acc = compute_accuracy_text(outputs, targets, self.vocab.stoi[PAD])
        self.log('val_acc', acc, prog_bar=True)
        self.log_loss("val", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 
    
    def test_step(self, batch, batch_idx):
        drawer_sequence, targets, dialogue, scene_before, scene_after, clip, mood, num_clip, topic = split_batch(batch)

        outputs, n, c, t, m = self(drawer_sequence, dialogue, scene_before, scene_after)
        
        outputs = outputs.to('cuda:0')
        targets = targets.to('cuda:0')
        
        loss, loss_n, loss_c, loss_t, loss_m = self.calculate_loss(outputs, targets, n, num_clip, c, clip, t, topic, m, mood)

        acc = compute_accuracy_text(outputs, targets, self.vocab.stoi[PAD])
        self.log('test_acc', acc, prog_bar=True)
        self.log_loss("test", loss, loss_n, loss_c, loss_t, loss_m)
        
        return loss + loss_n + loss_c + loss_t + loss_m 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
    
    def calculate_loss(self, outputs, targets, n, num_clip, c, clip, t, topic, m, mood):
        targets=targets.reshape(-1)
        outputs=outputs.view(-1, outputs.size(-1))
        
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in self.parameters())

        loss = self.loss_cross(outputs, targets) + l1_lambda * l1_norm
        loss_n = self.loss_cross_n(n, num_clip.argmax(dim=1)) if self.use_num_clip_decoder and n is not None else 0.0
        loss_c = self.loss_bce_c(c, clip.float()) if self.use_clipart_decoder and c is not None else 0.0
        loss_t = self.loss_bce_t(t, topic.float()) if self.use_topic_decoder and t is not None else 0.0
        loss_m = self.loss_bce_m(m, mood.float()) if self.use_mood_decoder and m is not None else 0.0
        
        return loss, loss_n, loss_c, loss_t, loss_m
        
    def log_loss(self, step, loss, loss_n, loss_c, loss_t, loss_m):
        self.log(f'{step}_loss', loss)
        self.log(f'{step}_loss_n', loss_n)
        self.log(f'{step}_loss_c', loss_c)
        self.log(f'{step}_loss_t', loss_t)
        self.log(f'{step}_loss_m', loss_m)
        return

    def reply(self, dialogue, scene_before, scene_after, use_sampling: bool, top_k: int, top_p: float, temperature: float, max_len = 39):
        word = torch.tensor(self.vocab.stoi[BOS]).view(1,-1).to(device)
        embeds = self.text_encoder(word)
        
        f_dialogue = self.dialogue_encoder(dialogue)
        f_scene_before = self.scene_compresser(self.scene_encoder(scene_before))
        f_scene_after = self.scene_compresser(self.scene_encoder(scene_after))
        features = torch.cat((f_dialogue, f_scene_before, f_scene_after), dim=1)  
        
        if self.use_num_clip_decoder:
            n = self.num_clip_decoder(features)
            features = torch.cat((features, n), dim=1)
        else:
            n = None
        if self.use_clipart_decoder:
            c = self.clipart_decoder(features)
            features = torch.cat((features, c), dim=1)
        else:
            c = None
        if self.use_topic_decoder:
            t = self.topic_decoder(features)
            features = torch.cat((features, t), dim=1)
        else:
            t = None
        if self.use_mood_decoder:
            m = self.mood_decoder(features)
            features = torch.cat((features, m), dim=1)
        else:
            m = None
        
        h, c = self.decoder.init_hidden_state(features)
        
        reply = []
        outputs = []
        outputs_tensor = torch.zeros(1,max_len, self.vocab_size).to(device)
        
        for i in range(max_len):
            attention_weighted_encoding, alpha = self.decoder.attention(features, h)
            gate = self.decoder.sigmoid(self.decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            lstm_input = torch.cat((embeds[:, 0], attention_weighted_encoding), dim=1)
            h, c = self.decoder.lstm_cell(lstm_input, (h, c))
            
            output = self.decoder.fcn(self.decoder.drop(h)).view(embeds.size(0),-1)
            outputs.append(output.view(-1, output.size(-1)))
            
            prediction_index = output.argmax(dim=1)
            reply.append(self.vocab.itos[prediction_index.item()])

            if self.vocab.itos[prediction_index.item()] == EOS:
                l = max_len - i -1
                outputs = torch.stack(outputs, dim = 1).to(device)
                outputs_tensor = torch.cat((outputs.clone().detach(), torch.zeros(1, l, self.vocab_size).to(device)), dim=1)
                break
            else:
                embeds = self.text_encoder(prediction_index.unsqueeze(0))

        return reply, outputs_tensor, n, c, t, m