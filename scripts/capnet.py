import torch
import torch.nn as nn
import numpy as np
import sys
import os
import json
import pickle
import pytorch_lightning as pl

from model.softgroup_module import SoftGroup
from model.proposal_module import ProposalModule
from model.caption_module import CaptionModule

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.config import CONF

vocab_path = os.path.join(CONF.PATH.DATA, "Scanrefer_vocabulary.json")
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")


class CapNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.vocabulary = json.load(open(vocab_path))
        self.embeddings = pickle.load(open(GLOVE_PICKLE, "rb"))

        # Define the model
        # -------------------------------------------------------------
        # ----------- SoftGroup-based Detection Backbone --------------
        self.softgroup_module = SoftGroup(in_channels=6,
                                          channels=32,
                                          num_blocks=7,
                                          semantic_classes=20,
                                          instance_classes=18,
                                          ignore_label=-100,
                                          grouping_cfg=CONF.grouping_cfg,
                                          instance_voxel_cfg=CONF.instance_voxel_cfg,
                                          train_cfg=CONF.train_cfg,
                                          test_cfg=CONF.test_cfg,
                                          fixed_modules=[])

        # --------------------- Proposal Module -----------------------
        self.proposal_module = ProposalModule(in_channels=32, out_channels=6, num_layers=2)

        # --------------------- Captioning Module ---------------------
        self.caption_module = CaptionModule(self.vocabulary, self.embeddings, emb_size=300, feat_size=32,
                                            hidden_size=512, num_proposals=256)

    def forward(self, batch):
        # semantic segmentation/grouping cluster/get cluster features
        batch['clus_feats_batch'], batch['select_feats'], batch['losses'], batch[
            'good_clu_masks'] = self.softgroup_module.forward(**batch)
        # predict bboxes
        self.proposal_module.forward(batch)
        # predict language_features
        self.caption_module.forward(batch)

        return batch

    def training_step(self, batch):
        batch = self.forward(batch)
        semantic_loss = batch['losses']['semantic_loss']
        offset_loss = batch['losses']['offset_loss']
        bbox_loss = batch['bbox_loss']
        cap_loss = batch['cap_loss']
        cap_acc = batch['cap_acc']
        print('sem_loss is: ', semantic_loss)
        print('offset_loss is: ', offset_loss)
        print('bbox_loss is: ', bbox_loss)
        print('cap_loss is: ', cap_loss)
        print('cap_acc is: ', cap_acc)
        loss = 10 * (semantic_loss + offset_loss) + 1 * bbox_loss + 0.1 * cap_loss
        return loss

    # def validation_step(self):
    #     return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
