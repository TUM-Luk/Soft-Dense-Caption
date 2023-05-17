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

from utils.nn_distance import nn_distance
from utils.box_util import box3d_iou_batch_tensor
from utils.val_helper import decode_caption, check_candidates, organize_candidates, prepare_corpus, compute_acc

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.config import CONF

vocab_path = os.path.join(CONF.PATH.DATA, "Scanrefer_vocabulary.json")
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")


class CapNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.vocabulary = json.load(open(vocab_path))
        self.embeddings = pickle.load(open(GLOVE_PICKLE, "rb"))
        self.organized = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_organized.json")))
        self.candidates = {}
        self.cap_acc = []
        self.cap_loss = []
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
                                            hidden_size=512, num_proposals=CONF.train_cfg.max_proposal_num)

    def forward_train(self, batch):
        # semantic segmentation/grouping cluster/get cluster features
        batch['clus_feats_batch'], batch['select_feats'], batch['losses'], batch[
            'good_clu_masks'] = self.softgroup_module.forward(**batch)
        # predict bboxes
        self.proposal_module.forward(batch)
        # predict language_features
        self.caption_module.forward(batch)

        return batch

    def forward_val(self, batch):
        # 新加的
        batch_size = batch['batch_size']
        num_proposals = CONF.train_cfg.max_proposal_num
        batch['clus_feats_batch'], batch['clus_center_batch'], batch['valid_clu_masks'], batch[
            'losses'] = self.softgroup_module.forward_val(
            **batch)
        # predict bboxes
        self.proposal_module.forward_val(batch)
        # predict language_features
        self.caption_module.forward(batch, use_tf=False, is_eval=True)
        captions = batch['lang_cap'].argmax(-1)
        gt_center = batch['center_label'][:, :, 0:3]
        _, batch['object_assignment'], _, _ = nn_distance(batch['clus_center_batch'], gt_center)

        # pick out object ids of detected objects
        detected_object_ids = torch.gather(batch["scene_object_ids"], 1, batch["object_assignment"])

        # bbox corners
        assigned_target_bbox_corners = torch.gather(
            batch["gt_box_corner_label"],
            1,
            batch["object_assignment"].view(batch_size, num_proposals, 1, 1).repeat(1, 1, 8, 3)
        )  # batch_size, num_proposals, 8, 3
        detected_bbox_corners = batch["bbox_corner"]  # batch_size, num_proposals, 8, 3

        # compute IoU between each detected box and each ground truth box
        ious = box3d_iou_batch_tensor(
            assigned_target_bbox_corners.view(-1, 8, 3),  # batch_size * num_proposals, 8, 3
            detected_bbox_corners.view(-1, 8, 3)  # batch_size * num_proposals, 8, 3
        ).view(batch_size, num_proposals)

        min_iou = 0
        # find good boxes (IoU > threshold)
        good_bbox_masks = ious > min_iou  # batch_size, num_proposals

        object_attn_masks = {}
        for batch_id in range(batch_size):
            scene_id = batch["scan_ids"][batch_id]
            object_attn_masks[scene_id] = np.zeros((num_proposals, num_proposals))
            for prop_id in range(num_proposals):
                if batch['valid_clu_masks'][batch_id, prop_id] == 1 and good_bbox_masks[batch_id, prop_id] == 1:
                    object_id = str(detected_object_ids[batch_id, prop_id].item())
                    caption_decoded = decode_caption(captions[batch_id, prop_id], self.vocabulary["idx2word"])

                    # print(scene_id, object_id)
                    try:
                        ann_list = list(self.organized[scene_id][object_id].keys())
                        object_name = self.organized[scene_id][object_id][ann_list[0]]["object_name"]

                        # store
                        key = "{}|{}|{}".format(scene_id, object_id, object_name)
                        # key = "{}|{}".format(scene_id, object_id)
                        self.candidates[key] = [caption_decoded]

                    except KeyError:
                        continue
        print('ceshi')
        print(self.candidates)

        return batch

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.candidates = {}  # 每次val开始时，重置candidates

        batch = self.forward_val(batch)
        semantic_loss = batch['losses']['semantic_loss']
        offset_loss = batch['losses']['offset_loss']
        print("validation test!!!!!!!!!")
        print('sem_loss is: ', semantic_loss)
        print('offset_loss is: ', offset_loss)

        return None

    def on_validation_end(self):
        # prepare corpus
        corpus_path = os.path.join(CONF.PATH.OUTPUT, "corpus_{}.json".format('val'))
        if not os.path.exists(corpus_path):
            print("preparing corpus...")
            raw_data = json.load(
                open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_{}.json".format('val'))))

            corpus = prepare_corpus(raw_data, CONF.TRAIN.MAX_DES_LEN)
            with open(corpus_path, "w") as f:
                json.dump(corpus, f, indent=4)
        else:
            print("loading corpus...")
            with open(corpus_path) as f:
                corpus = json.load(f)

        # check candidates
        # NOTE: make up the captions for the undetected object by "sos eos"
        candidates = check_candidates(corpus, self.candidates)
        candidates = organize_candidates(corpus, candidates)

        # save the predict captions
        pred_path = os.path.join(CONF.PATH.OUTPUT, "pred_{}.json".format('val'))

        with open(pred_path, "w") as f:
            json.dump(candidates, f, indent=4)

        print("computing scores...")

        bleu = capblue.Bleu(4).compute_score(corpus, candidates)
        # cider = capcider.Cider().compute_score(corpus, candidates)
        # rouge = caprouge.Rouge().compute_score(corpus, candidates)
        # meteor = capmeteor.Meteor().compute_score(corpus, candidates)
        print("validation数据集的bleu为：", bleu[0])
        return bleu

    def training_step(self, batch):
        batch = self.forward_train(batch)
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
        self.log("semantic_loss_train", semantic_loss, on_step=True, prog_bar=True, logger=True)
        loss = 10 * (semantic_loss + offset_loss) + 1 * bbox_loss + 0.1 * cap_loss
        return loss

    def on_train_epoch_end(self):
        filepath = f'model_checkpoint_epoch{self.current_epoch}.ckpt'
        self.trainer.save_checkpoint(filepath)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

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
