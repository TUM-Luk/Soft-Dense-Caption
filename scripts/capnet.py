import torch
import torch.nn as nn
import numpy as np
import sys
import os
import json
import pickle
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from model.caption_module import CaptionModule
from model.softgroup import SoftGroup

from model.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                              evaluate_semantic_acc, evaluate_semantic_miou)
from utils.nn_distance import nn_distance
from utils.box_util import box3d_iou_batch_tensor

from utils.val_helper import decode_caption, check_candidates, organize_candidates, prepare_corpus, compute_acc, \
    collect_results_cpu

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

from lib.capeval.bleu.bleu_scorer import BleuScorer

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.config import CONF

vocab_path = os.path.join(CONF.PATH.DATA, "Scanrefer_vocabulary.json")
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")


class CapNet(pl.LightningModule):
    def __init__(self, val_tf_on=False):
        super().__init__()
        self.writer = SummaryWriter("logs/")  # 指定TensorBoard日志保存路径
        self.results = []
        self.vocabulary = json.load(open(vocab_path))
        self.embeddings = pickle.load(open(GLOVE_PICKLE, "rb"))
        self.organized = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_organized.json")))
        self.candidates = {}
        self.val_tf_on = val_tf_on
        self.CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                        'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                        'bathtub', 'otherfurniture')
        # Define the model
        # -------------------------------------------------------------
        # ----------- SoftGroup-based Detection Backbone --------------
        self.softgroup_module = SoftGroup(in_channels=3,
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

        # # --------------------- Proposal Module -----------------------
        # self.proposal_module = ProposalModule(in_channels=32, out_channels=6, num_layers=2)

        # --------------------- Captioning Module ---------------------
        self.caption_module = CaptionModule(self.vocabulary, self.embeddings, emb_size=300, feat_size=32,
                                            hidden_size=512, num_proposals=CONF.train_cfg.max_proposal_num)

    def forward_train(self, batch):
        (loss, log_vars), batch['select_feats'], batch['good_clu_masks'], batch['object_feats'], batch[
            'object_mask'] = self.softgroup_module.forward(batch, return_loss=True)

        # predict language_features
        batch = self.caption_module.forward(batch)

        return loss, log_vars, batch

    def training_step(self, batch):
        loss, log_vars, batch = self.forward_train(batch)
        semantic_loss = log_vars['semantic_loss']
        offset_loss = log_vars['offset_loss']
        cls_loss = log_vars['cls_loss']
        mask_loss = log_vars['mask_loss']
        iou_score_loss = log_vars['iou_score_loss']
        cap_loss = batch['cap_loss']
        cap_acc = batch['cap_acc']
        loss = loss + 0.1 * cap_loss

        self.log("semantic_loss_train", semantic_loss, on_step=True, prog_bar=True, logger=True)
        self.log("offset_loss", offset_loss, on_step=True, prog_bar=True, logger=True)
        self.log("cls_loss", cls_loss, on_step=True, prog_bar=True, logger=True)
        self.log("mask_loss", mask_loss, on_step=True, prog_bar=True, logger=True)
        self.log("iou_score_loss", iou_score_loss, on_step=True, prog_bar=True, logger=True)
        self.log("cap_loss", cap_loss, on_step=True, prog_bar=True, logger=True)
        self.log("cap_acc", cap_acc, on_step=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):
        filepath = f'model0523_pretrain_attention_decay{self.current_epoch}.ckpt'
        self.trainer.save_checkpoint(filepath)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.candidates = {}
        if self.val_tf_on:
            loss, log_vars, batch = self.forward_train(batch)
            captions = batch['lang_cap'].argmax(-1)
            for i in range(batch['batch_size']):
                caption_decoded = decode_caption(captions[i], self.vocabulary["idx2word"])

            semantic_loss = log_vars['semantic_loss']
            offset_loss = log_vars['offset_loss']
            cls_loss = log_vars['cls_loss']
            mask_loss = log_vars['mask_loss']
            iou_score_loss = log_vars['iou_score_loss']
            cap_loss = batch['cap_loss']
            cap_acc = batch['cap_acc']
            self.log("val_semantic_loss_train", semantic_loss, on_step=True, prog_bar=True, logger=True)
            self.log("val_offset_loss", offset_loss, on_step=True, prog_bar=True, logger=True)
            self.log("val_cls_loss", cls_loss, on_step=True, prog_bar=True, logger=True)
            self.log("val_mask_loss", mask_loss, on_step=True, prog_bar=True, logger=True)
            self.log("val_iou_score_loss", iou_score_loss, on_step=True, prog_bar=True, logger=True)
            self.log("val_cap_loss", cap_loss, on_step=True, prog_bar=True, logger=True)
            self.log("val_cap_acc", cap_acc, on_step=True, prog_bar=True, logger=True)

        if not self.val_tf_on:
            (loss, log_vars), batch['select_feats'], batch['good_clu_masks'], batch['object_feats'], batch[
                'object_mask'] = self.softgroup_module.forward(batch, return_loss=True)

            # predict language_features without tf
            batch = self.caption_module.forward_sample_val(batch, use_tf=False)

            captions = batch['lang_cap'].argmax(-1)

            for i in range(batch['batch_size']):
                caption_decoded = decode_caption(captions[i], self.vocabulary["idx2word"])
                scene_id = str(batch['scan_ids'][i])
                object_id = str(int(batch['object_id'][i]))
                try:
                    ann_list = list(self.organized[scene_id][object_id].keys())
                    object_name = self.organized[scene_id][object_id][ann_list[0]]["object_name"]
                    # store
                    key = "{}|{}|{}".format(scene_id, object_id, object_name)
                    self.candidates[key] = [caption_decoded]

                except KeyError:
                    continue
        return None

    def on_validation_end(self):
        corpus_path = os.path.join(CONF.PATH.OUTPUT,"corpus_val.json")
        if not os.path.exists(corpus_path):
            print("preparing corpus...")
            raw_data = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
            corpus = prepare_corpus(raw_data, CONF.TRAIN.MAX_DES_LEN)
            with open(corpus_path, "w") as f:
                json.dump(corpus, f, indent=4)
        else:
            print("loading corpus...")
            with open(corpus_path) as f:
                corpus = json.load(f)

        self.candidates = check_candidates(corpus, self.candidates)
        self.candidates = organize_candidates(corpus, self.candidates)

        pred_path = os.path.join(CONF.PATH.OUTPUT, "pred_val.json")
        print("generating descriptions...")
        with open(pred_path, "w") as f:
            json.dump(self.candidates, f, indent=4)

        print("computing scores...")
        bleu = capblue.Bleu(4).compute_score(corpus, self.candidates)
        cider = capcider.Cider().compute_score(corpus, self.candidates)
        rouge = caprouge.Rouge().compute_score(corpus, self.candidates)
        self.logger.experiment.add_scalar("bleu-1", bleu[0][0],global_step=self.current_epoch)
        self.logger.experiment.add_scalar("bleu-2", bleu[0][1],global_step=self.current_epoch)
        self.logger.experiment.add_scalar("bleu-3", bleu[0][2],global_step=self.current_epoch)
        self.logger.experiment.add_scalar("bleu-4", bleu[0][3],global_step=self.current_epoch)
        self.logger.experiment.add_scalar("cider", cider[0],global_step=self.current_epoch)
        self.logger.experiment.add_scalar("rouge", rouge[0],global_step=self.current_epoch)

        print(bleu[0])
        print(cider[0])
        print(rouge[0])

        return None

    def visualization_softgroup(self, batch):
        # semantic preds/instance preds
        result = self.softgroup_module.forward_visualization(**batch)

        return result

    def visualization_proposal(self, input):
        # predict bboxes
        result = self.proposal_module.forward_visualization(input)

        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        result = dict()

        # generate senmantic preds results via softgroup.visualization
        result_softgroup = self.visualization_softgroup(batch)
        result.update(result_softgroup)

        # generate bbox preds results via softgroup.visualization
        result_proposal = self.visualization_proposal(result_softgroup)
        result.update(dict(pred_bboxes=result_proposal["pred_bboxes"]))

        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        return optimizer

    def test_step(self, batch, batch_idx):
        return None

# 单个val（用于debug)
#     def validation_step(self, batch, batch_idx):
#         if batch_idx == 0:
#             self.cap_acc = []
#         if batch_idx == 0:
#             self.cap_loss = []
#         batch = self.forward_train(batch)
#         semantic_loss = batch['losses']['semantic_loss']
#         offset_loss = batch['losses']['offset_loss']
#         bbox_loss = batch['bbox_loss']
#         cap_loss = batch['cap_loss']
#         cap_acc = batch['cap_acc']
#         print("validation test!!!!!!!!!")
#         print('sem_loss is: ', semantic_loss)
#         print('offset_loss is: ', offset_loss)
#         print('bbox_loss is: ', bbox_loss)
#         print('cap_loss is: ', cap_loss)
#         print('cap_acc is: ', cap_acc)
#         self.cap_acc.append(cap_acc)
#         self.cap_loss.append(cap_loss)
#         return None
#
#     def on_validation_end(self):
#         print("average cap_loss is:", sum(self.cap_loss) / 10)
#         print("average cap_acc is:", sum(self.cap_acc) / 10)
