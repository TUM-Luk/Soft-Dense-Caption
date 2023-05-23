import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.config import CONF
from utils.box_util import box3d_iou_batch_tensor

# constants
DC = ScannetDatasetConfig()


def select_target(data_dict):
    # predicted bbox
    pred_bbox = data_dict["bbox_corner"]  # batch_size, num_proposals, 8, 3
    batch_size, num_proposals, _, _ = pred_bbox.shape

    # ground truth bbox
    gt_bbox = data_dict["ref_box_corner_label"]  # batch_size, 8, 3

    target_ids = []
    target_ious = []
    for i in range(batch_size):
        # convert the bbox parameters to bbox corners
        pred_bbox_batch = pred_bbox[i]  # num_proposals, 8, 3
        gt_bbox_batch = gt_bbox[i].unsqueeze(0).repeat(num_proposals, 1, 1)  # num_proposals, 8, 3
        ious = box3d_iou_batch_tensor(pred_bbox_batch, gt_bbox_batch)
        target_id = ious.argmax().item()  # 0 ~ num_proposals - 1
        target_ids.append(target_id)
        target_ious.append(ious[target_id])

    target_ids = torch.LongTensor(target_ids).cuda()  # batch_size
    target_ious = torch.FloatTensor(target_ious).cuda()  # batch_size

    return target_ids, target_ious


class CaptionModule(nn.Module):
    def __init__(self, vocabulary, embeddings, emb_size=300, feat_size=32, hidden_size=512, num_proposals=128):
        super().__init__()

        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.num_vocabs = len(vocabulary["word2idx"])

        self.emb_size = emb_size
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.num_proposals = num_proposals

        # attention layer
        self.attend = nn.Linear(feat_size, 1, bias=False)

        # transform the object_feature to higher_dimension
        self.map_feat = nn.Sequential(
            nn.Linear(feat_size, emb_size),
            nn.ReLU()
        )

        # captioning core 主体部分
        self.recurrent_cell = nn.LSTMCell(
            input_size=emb_size + feat_size + feat_size,  # word vector + clus feature + avg_obj_feature
            hidden_size=hidden_size,
        )
        # 输出分类层
        self.classifier = nn.Linear(hidden_size, self.num_vocabs)

    def step(self, step_input, hidden, cell):
        (hidden, cell) = self.recurrent_cell(step_input, (hidden, cell))  # num_proposals, hidden_size

        return hidden, hidden, cell

    def compute_loss(self, data_dict):
        pred_caps = data_dict['lang_cap']  # (B, num_words - 1, num_vocabs)
        num_words = data_dict['lang_len'].max()
        target_caps = data_dict['lang_ids'][:, 1:num_words]  # (B, num_words - 1)

        _, _, num_vocabs = pred_caps.shape

        # caption loss
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        cap_loss = criterion(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1))

        # mask out bad boxes
        good_bbox_masks = data_dict["good_clu_masks"].unsqueeze(1).repeat(1, num_words - 1)  # (B, num_words - 1)
        good_bbox_masks = good_bbox_masks.reshape(-1)  # (B * num_words - 1)
        cap_loss = torch.sum(cap_loss * good_bbox_masks) / (torch.sum(good_bbox_masks) + 1e-6)

        num_good_bbox = data_dict["good_clu_masks"].sum()
        if num_good_bbox > 0:  # only apply loss on the good boxes
            pred_caps = pred_caps[data_dict["good_clu_masks"]]  # num_good_bbox
            target_caps = target_caps[data_dict["good_clu_masks"]]  # num_good_bbox

            # caption acc
            pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1)  # num_good_bbox * (num_words - 1)
            target_caps = target_caps.reshape(-1)  # num_good_bbox * (num_words - 1)
            masks = target_caps != 0
            masked_pred_caps = pred_caps[masks]
            masked_target_caps = target_caps[masks]
            cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()
        else:  # zero placeholder if there is no good box
            cap_acc = torch.zeros(1)[0].cuda()

        return cap_loss, cap_acc

    def forward(self, data_dict, use_tf=True, is_eval=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        if not is_eval:
            data_dict = self.forward_sample_batch(data_dict, max_len)
            data_dict['cap_loss'], data_dict['cap_acc'] = self.compute_loss(data_dict)
        else:
            data_dict = self.forward_scene_batch(data_dict, use_tf, max_len)

        return data_dict

    def forward_sample_batch(self, data_dict, max_len=CONF.TRAIN.MAX_DES_LEN, min_iou=CONF.TRAIN.MIN_IOU_THRESHOLD):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"].cuda()  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"].cuda()  # batch_size
        obj_feats = data_dict["select_feats"].cuda()  # batch_size, feat_size
        all_obj_feats = data_dict['object_feats'].cuda()  # batch_size, num_proposal, feat_size
        object_mask = data_dict['object_mask'].cuda()  # batch_size, num_proposal

        object_mask = torch.unsqueeze(object_mask, dim=2)  # batch_size, num_proposal, 1

        # avg_obj_feats = torch.sum(all_obj_feats * object_mask, dim=1)  # batch_size, feat_size
        # avg_obj_feats = avg_obj_feats / (torch.sum(object_mask, dim=1) + 1e-6)  # batch_size, feat_size

        combined = all_obj_feats + obj_feats.unsqueeze(dim=1)  # batch_size, num_proposal, feat_size
        combined = torch.tanh(combined)
        scores = self.attend(combined)  # batch_size, num_proposal, 1
        scores.masked_fill_(object_mask == 0, float('-1e30'))

        masks = F.softmax(scores, dim=1)  # batch_size, num_proposal, 1
        attended = all_obj_feats * masks
        attended = attended.sum(1)  # batch_size, feat_size

        num_words = des_lens.max()
        batch_size = des_lens.shape[0]

        # transform the features
        # obj_feats = self.map_feat(obj_feats)  # batch_size, emb_size
        # obj_feats = obj_feats.cuda()

        # recurrent from 0 to max_len - 2
        outputs = []
        hidden = torch.zeros(batch_size, self.hidden_size).cuda()  # batch_size, hidden_size
        cell = torch.zeros(batch_size, self.hidden_size).cuda()  # batch_size, hidden_size
        step_id = 0
        step_input = torch.concat((word_embs[:, step_id], obj_feats, attended),
                                  dim=1)  # batch_size, input_size(300+32+32)
        while True:
            # feed
            step_output, hidden, cell = self.step(step_input, hidden, cell)
            step_output = self.classifier(step_output)  # batch_size, num_vocabs

            # store
            step_output = step_output.unsqueeze(1)  # batch_size, 1, num_vocabs
            outputs.append(step_output)

            # next step
            step_id += 1
            if step_id == num_words - 1:
                break  # exit for train mode
            step_input = torch.concat((word_embs[:, step_id], obj_feats, attended),
                                      dim=1)  # batch_size, input_size

        outputs = torch.cat(outputs, dim=1)  # batch_size, num_words - 1/max_len, num_vocabs

        # store
        data_dict["lang_cap"] = outputs

        return data_dict

    def forward_sample_val(self, data_dict, use_tf=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        # unpack
        word_embs = data_dict["lang_feat"].cuda()  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"].cuda()  # batch_size
        obj_feats = data_dict["select_feats"].cuda()  # batch_size, feat_size
        all_obj_feats = data_dict['object_feats'].cuda()  # batch_size, num_proposal, feat_size
        object_mask = data_dict['object_mask'].cuda()  # batch_size, num_proposal

        object_mask = torch.unsqueeze(object_mask, dim=2)  # batch_size, num_proposal, 1

        # avg_obj_feats = torch.sum(all_obj_feats * object_mask, dim=1)  # batch_size, feat_size
        # avg_obj_feats = avg_obj_feats / (torch.sum(object_mask, dim=1) + 1e-6)  # batch_size, feat_size

        combined = all_obj_feats + obj_feats.unsqueeze(dim=1)  # batch_size, num_proposal, feat_size
        combined = torch.tanh(combined)
        scores = self.attend(combined)  # batch_size, num_proposal, 1
        scores.masked_fill_(object_mask == 0, float('-1e30'))

        masks = F.softmax(scores, dim=1)  # batch_size, num_proposal, 1
        attended = all_obj_feats * masks
        attended = attended.sum(1)  # batch_size, feat_size

        num_words = des_lens.max()
        batch_size = des_lens.shape[0]

        # transform the features
        # obj_feats = self.map_feat(obj_feats)  # batch_size, emb_size
        # obj_feats = obj_feats.cuda()

        # recurrent from 0 to max_len - 2
        outputs = []
        hidden = torch.zeros(batch_size, self.hidden_size).cuda()  # batch_size, hidden_size
        cell = torch.zeros(batch_size, self.hidden_size).cuda()  # batch_size, hidden_size
        step_id = 0
        step_input = torch.concat((word_embs[:, step_id], obj_feats, attended), dim=1)  # batch_size, input_size
        while True:
            # feed
            step_output, hidden, cell = self.step(step_input, hidden, cell)
            step_output = self.classifier(step_output)  # batch_size, num_vocabs

            # predicted word
            step_preds = []
            for batch_id in range(batch_size):
                idx = step_output[batch_id].argmax()  # 0 ~ num_vocabs
                word = self.vocabulary["idx2word"][str(idx.item())]
                emb = torch.FloatTensor(self.embeddings[word]).unsqueeze(0).cuda()  # 1, emb_size
                step_preds.append(emb)

            step_preds = torch.cat(step_preds, dim=0)  # batch_size, emb_size

            # store
            step_output = step_output.unsqueeze(1)  # batch_size, 1, num_vocabs
            outputs.append(step_output)

            # next step
            step_id += 1
            if use_tf and step_id == num_words - 1:
                break  # exit for tf_val mode
            if not use_tf and step_id == max_len - 1:
                break  # exit for no_tf_val mode

            if use_tf:
                step_input = torch.concat((word_embs[:, step_id], obj_feats, attended),
                                          dim=1)  # batch_size, input_size
            if not use_tf:
                step_input = torch.concat((step_preds, obj_feats, attended), dim=1)  # batch_size, input_size

        outputs = torch.cat(outputs, dim=1)  # batch_size, num_words - 1/max_len, num_vocabs
        data_dict["lang_cap"] = outputs

        return data_dict

    def forward_scene_batch(self, data_dict, use_tf=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"]  # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"]  # batch_size
        obj_feats = data_dict["clus_feats_batch"]  # batch_size, num_proposals, feat_size

        num_words = des_lens.max()
        batch_size = des_lens.shape[0]

        # transform the features
        obj_feats = self.map_feat(obj_feats)  # batch_size, num_proposals, emb_size

        # recurrent from 0 to max_len - 2
        outputs = []
        for prop_id in range(self.num_proposals):
            # select object features
            target_feats = obj_feats[:, prop_id]  # batch_size, emb_size

            # start recurrence
            prop_outputs = []
            hidden = target_feats  # batch_size, emb_size
            cell = torch.zeros(batch_size, self.emb_size).cuda()
            step_id = 0
            step_input = word_embs[:, step_id]  # batch_size, emb_size 一开始即soc的embedding
            while True:
                # feed
                step_output, hidden, cell = self.step(step_input, hidden, cell)
                step_output = self.classifier(step_output)  # batch_size, num_vocabs

                # predicted word
                step_preds = []
                for batch_id in range(batch_size):
                    idx = step_output[batch_id].argmax()  # 0 ~ num_vocabs
                    word = self.vocabulary["idx2word"][str(idx.item())]
                    emb = torch.FloatTensor(self.embeddings[word]).unsqueeze(0).cuda()  # 1, emb_size
                    step_preds.append(emb)

                step_preds = torch.cat(step_preds, dim=0)  # batch_size, emb_size

                # store
                step_output = step_output.unsqueeze(1)  # batch_size, 1, num_vocabs
                prop_outputs.append(step_output)

                # next step
                step_id += 1
                if not use_tf and step_id == max_len - 1: break  # exit for eval mode
                if use_tf and step_id == num_words - 1: break  # exit for train mode
                step_input = step_preds if not use_tf else word_embs[:, step_id]  # batch_size, emb_size

            prop_outputs = torch.cat(prop_outputs, dim=1).unsqueeze(
                1)  # batch_size, 1, num_words - 1/max_len, num_vocabs
            outputs.append(prop_outputs)

        outputs = torch.cat(outputs, dim=1)  # batch_size, num_proposals, num_words - 1/max_len, num_vocabs

        # store
        data_dict["lang_cap"] = outputs

        return data_dict
