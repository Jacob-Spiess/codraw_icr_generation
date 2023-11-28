#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the experiment with a PytorchLightining model.
"""

from argparse import Namespace
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F

from icr.aux import mask_pads #filter_checkpoint
from icr.constants import REDUCTION, SPLITS
from icr.evaluator import Metrics, Outputs
from icr import constants
from icr.models import TopicModel, ICRModel, TextEncoder, TextDecoder, EmbeddingCompresser, ProbDecoder
from icr.components import (
    ActionsMaker, CrossEncoder, iCRClipDecoder, iCRTurnDecoder,
    SceneEncoder, SelfCrossEncoder, StateEmbedding, LearnableLossWeights)

class iCRGeneration(pl.LightningModule):
    """Lightining experiment."""
    def __init__(self, generation_config: Dict[str, int],  model_config: Dict[str, int]):
        super().__init__()
        
        self.config = generation_config

        self.gallery_embedder = StateEmbedding(n_cliparts=constants.N_CLIPS, n_faces=constants.N_FACE_CLASSES,
                                                n_flips=constants.N_FLIP_CLASSES, n_poses=constants.N_POSE_CLASSES,
                                                n_positions=constants.LEN_POSITION, n_presences=constants.N_PRESENCE_CLASSES,
                                                n_sizes=constants.N_SIZE_CLASSES, total_dim=model_config["d_model"],
                                                prefix=constants.BEFORE_PREFIX)
        self.positional_encoding = PositionalEncoding1D(model_config["d_model"])
        # which input components to use besides gallery
        self.text_compresser = nn.Linear(768, model_config["d_model"])
        self.scene_encoder = SceneEncoder(d_model=model_config["d_model"], dropout=model_config["dropout"],
                                            unfreeze=model_config["unfreeze_resnet"],
                                            dont_preprocess_scenes=model_config["dont_preprocess_scenes"])
        # contextual encoder of the cliparts
        self.encoder = SelfCrossEncoder(d_model=model_config["d_model"], hidden_dim=model_config["hidden_dim_trf"],
            nheads=model_config["nheads"], nlayers=model_config["nlayers"], dropout=model_config["dropout"])        
        
        
        self.topicmodel = TopicModel(model_config)
        self.icrmodel = ICRModel(model_config)

        
        
        #self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(generation_config["pos_weight"]), reduction=REDUCTION)
        self.loss = nn.CrossEntropyLoss()
        self.loss_weights = LearnableLossWeights(self.icrmodel.labels)

        self.evaluator = self._define_metrics()
        self.inputs = self._define_inputs()
        self.outputs = Outputs(generation_config["outputs_path"], self.icrmodel.labels)

    def _define_metrics(self):
        metrics = {f'{split}-metrics': Metrics(self.icrmodel.labels, split) for split in SPLITS}
        return nn.ModuleDict(metrics)

    def _compute_loss(self, outputs: Dict[str, Tensor], gold: Dict[str, Tensor], split: str) -> Tensor:
        loss_sum = 0
        for name, preds in outputs.items():
            gold_labels = gold[name.replace('pred-', '')].float()
            loss = self.loss(preds, gold_labels)
            loss_name = f'{split}_{name}_loss'
            self.log(loss_name, loss, on_step=False, on_epoch=True)
            if self.config["use_weighted_loss"]:
                loss = self.loss_weights(name, loss)
            loss_sum += loss
        self.log(f'{split}_loss', loss_sum, on_step=False, on_epoch=True)
        return loss_sum

    def configure_optimizers(self) -> List:
        optimizer = Adam(self.parameters(), lr=self.config["lr"])
        return optimizer
    
    def filter_batch(self, batch: Dict[str, Tensor]) -> Tuple[Dict, Dict]:
        """Extract from the batch dict only what the model uses."""
        inputs = {k: v for k, v in batch.items() if k in self.inputs}
        labels_topic = {k: v for k, v in batch.items() if k in self.topicmodel.labels}
        labels_icr = {k: v for k, v in batch.items() if k in self.icrmodel.labels}

        return inputs, labels_topic, labels_icr
    
    def _define_inputs(self) -> Tuple[List, List]:
        """Create a list of input and label names used in the model."""
        inputs = self.gallery_embedder.inputs + ['dialogue', 'scene_before', 'scene_after']
        return inputs


    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        
        # state of the gallery and symbolic scene before the actions
        gallery = self.gallery_embedder(inputs)

        memory = []
        for scene_key in ['scene_before', 'scene_after']:
            scene_emb = self.scene_encoder(inputs[scene_key])
            memory.append(scene_emb)
            
        if memory:
            # add position encodings to elements in the image features
            # although it has its own learnt position encodings, we also
            # want to capture its position in the memory
            scenes = torch.cat(memory, dim=1)
            positions = self.positional_encoding(scenes)
            memory = [scenes + positions]

        dialogue = self.text_compresser(inputs['dialogue'])
        positions = mask_pads(dialogue, self.positional_encoding(dialogue))
        memory.append(dialogue + positions)

        memory = torch.cat(memory, dim=1)
        game_state = self.encoder(gallery, memory)        
        
       
        pred_icr_clip = self.topicmodel(game_state)
        enriched_gamestate = torch.cat((game_state, pred_icr_clip), dim=1)
        pred_icr = self.icrmodel(enriched_gamestate)
        return pred_icr, pred_icr_clip


    def training_step(self, batch: Dict[str, Tensor], batch_idx) -> Tensor:
        inputs, labels_topic, labels_icr = self.filter_batch(batch)
        outputs = self(inputs, labels_icr)
        loss = self._compute_loss(outputs, labels_icr, 'train')
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch: Dict[str, Tensor], batch_idx) -> None:
        inputs, labels_topic, labels_icr = self.filter_batch(batch)
        outputs = self(inputs)
        loss = self._compute_loss(outputs, labels_icr, 'val')
        self.log('val_loss', loss)
        
    def test_step(self, batch: Dict[str, Tensor], batch_idx) -> None:
        inputs, labels_topic, labels_icr = self.filter_batch(batch)
        outputs = self(inputs)
        loss = self._compute_loss(outputs, labels_icr, 'test')
        self.log('test_loss', loss)


class ICRModel(pl.LightningModule):
    def __init__(self, vocab, config):
        super(ICRModel, self).__init__()
        
        vocab_size = len(vocab)
        max_length = vocab.max_token
        
        self.loss_fn = CrossEntropyLoss(reduction="sum")
        
        # Saving hyperparameters
        self.save_hyperparameters()

        # Sequence model
        self.se1 = nn.Embedding(vocab_size, config["d_model"], padding_idx=0)
        self.se2 = nn.Dropout(config["dropout"])
        self.se3 = nn.Linear(config["hidden_dim"], config["hidden_dim"]) 

        # Decoder model
        self.decoder1 = nn.LSTM(config["hidden_dim"], config["hidden_dim"])
        self.decoder2 = nn.Linear(config["hidden_dim"], vocab_size)

    def forward(self, input_sequence):

        # Sequence model
        se1_out = self.se1(input_sequence)
        se2_out = self.se2(se1_out)
        se3_out, _ = self.se3(se2_out)

        # Decoder model
        decoder1_out = se3_out[:,-1,:]   # Assuming you want to use the last output of the LSTM
        decoder2_out = F.relu(self.decoder1(decoder1_out))
        outputs = F.softmax(self.decoder2(decoder2_out), dim=1)

        return outputs
    
    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch["X"], batch["y"]
        
        y_pred = self(X_batch)
        loss = self.loss_fn(y_pred, y_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch["X"], batch["y"]

        y_pred = self(X_batch)
        loss = self.loss_fn(y_pred, y_batch)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        X_batch, y_batch = batch["X"], batch["y"]
        y_pred = self(X_batch)
        loss = self.loss_fn(y_pred, y_batch)

        # Log accuracy
        acc = (y_pred.argmax(dim=1) == y_batch).sum().item() / len(y_batch)
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    
    
class ICRModel1(pl.LightningModule):
    def __init__(self, vocab, config):
        super(ICRModel1, self).__init__()
        
        vocab_size = len(vocab)
        self.vocab_size = vocab_size
        max_length = vocab.max_token
        self.vocab = vocab

        self.loss_cross = CrossEntropyLoss()
        self.loss_bce = BCEWithLogitsLoss()

        self.save_hyperparameters()

        self.text_encoder = TextEncoder(vocab_size, config["hidden_dim"], config["hidden_dim"], config["dropout"])
        self.dialogue_encoder = EmbeddingCompresser(config["hidden_dim"])
        self.decoder = TextDecoder(config["hidden_dim"], config["hidden_dim"], vocab_size, config["nlayers"], config["dropout"])
        self.topic_decoder = ProbDecoder(config["hidden_dim"], config["hidden_dim"], 6 , config["dropout"])
        self.num_clip_decoder = ProbDecoder(config["hidden_dim"], config["hidden_dim"], 5, config["dropout"])
        self.mood_decoder = ProbDecoder(config["hidden_dim"], config["hidden_dim"], 7, config["dropout"])
        self.clipart_decoder = ProbDecoder(config["hidden_dim"], config["hidden_dim"], 28, config["dropout"])

    def forward(self, input_sequence, dialogue, clip, mood, num_clip, topic):
        
        features = self.dialogue_encoder(dialogue)
        #features = features.unsqueeze(1)       
        embeds = self.text_encoder(input_sequence)
        
        outputs_c = self.clipart_decoder(features)
        outputs_m = self.mood_decoder(features)
        outputs_n = self.num_clip_decoder(features)
        outputs_t = self.topic_decoder(features)


        encoding = torch.cat((features.unsqueeze(1), embeds), dim=1)
        outputs = self.decoder(encoding)
        
        return outputs, outputs_c, outputs_m, outputs_n, outputs_t
    
    def training_step(self, batch, batch_idx):
        drawer_sequence = batch["drawer_reply_tokenized"][:, :-1]
        targets =  batch["drawer_reply_tokenized"]#[:,1: ] 
        dialogue = batch["dialogue"]
        clip = batch["icr_clip_label"]
        mood = batch["icr_mood"]
        num_clip = batch["icr_num_clip"]
        topic = batch["icr_topic"]
        
        outputs, c, m, n, t = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        loss = self.loss_cross(outputs.view(-1,self.vocab_size), targets.reshape(-1))
        loss_c = self.loss_bce(c, clip.float())
        loss_m = self.loss_bce(m, mood.float())
        loss_n = self.loss_cross(n, num_clip.argmax(dim=1))
        loss_t = self.loss_bce(t, topic.float())

        self.log('train_loss', loss)
        self.log('train_loss_c', loss_c)
        self.log('train_loss_m', loss_m)
        self.log('train_loss_n', loss_n)
        self.log('train_loss_t', loss_t)
        return loss + loss_c + loss_m + loss_n + loss_t

    def validation_step(self, batch, batch_idx):
        drawer_sequence = batch["drawer_reply_tokenized"][:, :-1]
        targets =  batch["drawer_reply_tokenized"]#[:,1: ]  
        dialogue = batch["dialogue"]
        clip = batch["icr_clip_label"]
        mood = batch["icr_mood"]
        num_clip = batch["icr_num_clip"]
        topic = batch["icr_topic"]

        outputs, c, m, n, t = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)

        targets_loss = targets.reshape(-1)
        outputs_loss = outputs.view(-1, outputs.size(-1))
        loss = self.loss_cross(outputs_loss, targets_loss)        
        # mask = tarets_loss != 0 
        # acc = (outputs_loss[mask].argmax(dim=1) == targets_loss[mask]).float().mean().item()
        acc = (outputs_loss.argmax(dim=1) == targets_loss).sum().item() / len(targets_loss)
        
        loss_c = self.loss_bce(c, clip.float())
        loss_m = self.loss_bce(m, mood.float())
        loss_n = self.loss_cross(n, num_clip.argmax(dim=1))
        loss_t = self.loss_bce(t, topic.float())

        self.log('val_acc', acc, prog_bar=True)
        self.log('val_loss', loss)
        self.log('val_loss_c', loss_c)
        self.log('val_loss_m', loss_m)
        self.log('val_loss_n', loss_n)
        self.log('val_loss_t', loss_t)
        return loss + loss_c + loss_m + loss_n + loss_t
    
    def test_step(self, batch, batch_idx):
        drawer_sequence = batch["drawer_reply_tokenized"][:, :-1]
        targets =  batch["drawer_reply_tokenized"] 
        dialogue = batch["dialogue"]
        clip = batch["icr_clip_label"]
        mood = batch["icr_mood"]
        num_clip = batch["icr_num_clip"]
        topic = batch["icr_topic"]

        outputs, c, m, n, t = self(drawer_sequence, dialogue, clip, mood, num_clip, topic)
        
        targets=targets.reshape(-1)
        outputs=outputs.view(-1, outputs.size(-1))
        loss = self.loss_cross(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).sum().item() / len(targets)
        
        loss_c = self.loss_bce(c, clip.float())
        loss_m = self.loss_bce(m, mood.float())
        loss_n = self.loss_cross(n, num_clip.argmax(dim=1))
        loss_t = self.loss_bce(t, topic.float())

        self.log('test_loss', loss)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_loss_c', loss_c)
        self.log('test_loss_m', loss_m)
        self.log('test_loss_n', loss_n)
        self.log('test_loss_t', loss_t)
        return loss + loss_c + loss_m + loss_n + loss_t

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
    
    def calculate_loss(self, outputs, targets):
        #loss = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the index for padding
        #return loss(outputs.view(-1, outputs.size(2)), targets.view(-1))
        outputs_2d = outputs.view(-1, outputs.size(-1))

        # Reshape targets to 1D tensor
        targets_1d = targets.view(-1)

        loss = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the index for padding
        return loss(outputs_2d, targets_1d)

    def reply(self, dialogue, vocab, max_len = 20, hidden=None):
        word = torch.tensor(self.vocab.stoi[BOS]).view(1,-1)
        features = self.dialogue_encoder(dialogue)
        features = features.unsqueeze(1)

        embeds = self.text_encoder(word)
        encoding = torch.cat((features, embeds), dim=1)

        reply = []

        for _ in range(max_len):
            x, hidden = self.decoder.decoder1(encoding, hidden)
            output = self.decoder.decoder2(x).view(encoding.size(0),-1)

            prediction_index = output.argmax(dim=1)
            reply.append(vocab.itos[prediction_index.item()])

            if vocab.itos[prediction_index.item()] == EOS:
                break
            else:
                features = self.embedding(prediction_index.unsqueeze(0))

        return reply
