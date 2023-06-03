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

from utils.val_helper import decode_caption, check_candidates, organize_candidates, prepare_corpus, collect_results_cpu, \
    save_pred_instances, save_gt_instances

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

from lib.capeval.bleu.bleu_scorer import BleuScorer

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.config import CONF

from plyfile import PlyData, PlyElement
from utils.box_util import get_3d_box_batch
import glob
from multiprocessing import Pool
from eval_det import eval_det
from visualization import write_bbox

vocab_path = os.path.join(CONF.PATH.DATA, "Scanrefer_vocabulary.json")
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")


class CapNet(pl.LightningModule):
    def __init__(self, val_tf_on=False):
        super().__init__()
        self.writer = SummaryWriter("logs/")  # 指定TensorBoard日志保存路径
        self.results = []  # 用来验证map
        self.vocabulary = json.load(open(vocab_path))
        self.embeddings = pickle.load(open(GLOVE_PICKLE, "rb"))
        self.organized = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_organized.json")))
        self.candidates = {}
        self.val_tf_on = val_tf_on
        self.CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                        'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                        'bathtub', 'otherfurniture')
        self.nyu_id = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

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
        loss = loss + cap_loss

        self.log("semantic_loss_train", semantic_loss, on_step=True, prog_bar=True, logger=True)
        self.log("offset_loss", offset_loss, on_step=True, prog_bar=True, logger=True)
        self.log("cls_loss", cls_loss, on_step=True, prog_bar=True, logger=True)
        self.log("mask_loss", mask_loss, on_step=True, prog_bar=True, logger=True)
        self.log("iou_score_loss", iou_score_loss, on_step=True, prog_bar=True, logger=True)
        self.log("cap_loss", cap_loss, on_step=True, prog_bar=True, logger=True)
        self.log("cap_acc", cap_acc, on_step=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):
        filepath = f'model0528_e2e_{self.current_epoch}.ckpt'
        self.trainer.save_checkpoint(filepath)
        return None

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
        corpus_path = os.path.join(CONF.PATH.OUTPUT, "corpus_val.json")
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

        self.logger.experiment.add_scalar("bleu-1", bleu[0][0], global_step=self.current_epoch)
        self.logger.experiment.add_scalar("bleu-2", bleu[0][1], global_step=self.current_epoch)
        self.logger.experiment.add_scalar("bleu-3", bleu[0][2], global_step=self.current_epoch)
        self.logger.experiment.add_scalar("bleu-4", bleu[0][3], global_step=self.current_epoch)
        self.logger.experiment.add_scalar("cider", cider[0], global_step=self.current_epoch)
        self.logger.experiment.add_scalar("rouge", rouge[0], global_step=self.current_epoch)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def test_step(self, batch, batch_idx):  # batch size must be 1
        scan_id = batch['scan_ids'][0]
        # torch.save((batch['coords_float'], batch['feats'], batch['semantic_labels'], batch['instance_labels']),
        #            os.path.join(CONF.PATH.BASE, "outputs", scan_id + '_inst_nostuff.pth'))

        ret = self.softgroup_module.forward(batch, return_loss=False)
        print(len(ret['pred_instances']))
        self.results.append(ret)
        return None

    def on_test_end(self):
        self.results = collect_results_cpu(self.results, len(self.results))

        scan_ids = []
        pred_insts, gt_insts = [], []
        for res in self.results:
            scan_ids.append(res['scan_id'])
            pred_insts.append(res['pred_instances'])
            gt_insts.append(res['gt_instances'])
        root = CONF.PATH.OUTPUT
        save_pred_instances(root, 'pred_instance', scan_ids, pred_insts, self.nyu_id)
        save_gt_instances(root, 'gt_instance', scan_ids, gt_insts, self.nyu_id)

        # calculate MAP
        data_path = CONF.PATH.SCANNET
        results_path = CONF.PATH.OUTPUT
        iou_threshold = 0.5  # adjust threshold here
        instance_paths = glob.glob(os.path.join(results_path, 'pred_instance', '*.txt'))
        instance_paths.sort()

        CLASS_LABELS = [
            'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
            'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
            'otherfurniture'
        ]
        VALID_CLASS_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

        def single_process(instance_path):
            img_id = os.path.basename(instance_path)[:-4]
            print('Processing', img_id)
            gt = os.path.join(data_path, 'val', img_id + '_inst_nostuff.pth')  # 0-based index
            assert os.path.isfile(gt)
            coords, rgb, semantic_label, instance_label,_,_,_,_ = torch.load(gt)
            # coords = coords.cpu().numpy()
            # semantic_label = semantic_label.cpu().numpy()
            # instance_label = instance_label.cpu().numpy()
            pred_infos = open(instance_path, 'r').readlines()
            pred_infos = [x.rstrip().split() for x in pred_infos]  # nyu_id index
            mask_path, labels, scores = list(zip(*pred_infos))
            pred = []
            for mask_path, label, score in pred_infos:
                mask_full_path = os.path.join(results_path, 'pred_instance', mask_path)
                mask = np.array(open(mask_full_path).read().splitlines(), dtype=int).astype(bool)
                instance = coords[mask]
                box_min = instance.min(0)
                box_max = instance.max(0)
                box = np.concatenate([box_min, box_max])
                class_name = CLASS_LABELS[VALID_CLASS_IDS.index(int(label))]
                pred.append((class_name, box, float(score)))

            instance_num = int(instance_label.max()) + 1
            gt = []

            for i in range(instance_num):
                inds = instance_label == i
                gt_label_loc = np.nonzero(inds)[0][0]
                cls_id = int(semantic_label[gt_label_loc])
                if cls_id >= 2:
                    instance = coords[inds]
                    box_min = instance.min(0)
                    box_max = instance.max(0)
                    box = np.concatenate([box_min, box_max])
                    class_name = CLASS_LABELS[cls_id - 2]
                    gt.append((class_name, box))
            return img_id, pred, gt

        pred_gt_results = []
        for i in instance_paths:
            pred_gt_results.append(single_process(i))

        pred_all = {}
        gt_all = {}
        for img_id, pred, gt in pred_gt_results:
            pred_all[img_id] = pred
            gt_all[img_id] = gt

        # print(pred_all)
        # print(gt_all)

        print('Evaluating...')
        eval_res = eval_det(pred_all, gt_all, ovthresh=iou_threshold)
        aps = list(eval_res[-1].values())
        mAP = np.mean(aps)
        print('mAP:', mAP)

    # 计算语言指标的代码!!!
    # def test_step(self, batch, batch_idx):  # batch size must be 1
    #     if batch_idx == 0:
    #         self.candidates = {}
    #
    #     batch[
    #         'object_feats'], cls_scores, iou_scores, mask_scores, proposals_idx = self.softgroup_module.forward_scene_test(
    #         **batch)
    #     scan_id = batch['scan_ids'][0]
    #     num_points = batch['coords_float'].shape[0]  # num_point
    #     num_instances = cls_scores.size(0)  # proposal的数量
    #     cls_scores = cls_scores.softmax(1)  # softmax分类得分
    #     final_score = cls_scores * iou_scores.clamp(0, 1)  # M, 19,每个proposal分类的最终得分
    #     max_cls_score, final_cls = final_score.max(1)  # M, 1 和 M, 1 每个proposal得到的分类最高分，以及属于哪个类
    #     print(scan_id)
    #     print(cls_scores.max(1))
    #     print(max_cls_score)
    #     print(final_cls)
    #     # print(final_score)
    #     batch['final_feats'] = batch['object_feats']  # M, 32
    #     batch = self.caption_module.forward_scene_test(batch)
    #     captions = batch['final_lang'].argmax(-1)  # M , (max_len-1)
    #
    #     mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
    #     for i in range(num_instances):
    #         cur_mask_scores = mask_scores[:, final_cls[i]]  # N, 1， N个点对第i个proposal的类的mask score
    #         mask_inds = cur_mask_scores > -0.5  # threshold 取mask高于阈值的点
    #         cur_proposals_idx = proposals_idx[mask_inds].long()
    #         mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1  # M , num_point 表示有哪些点可能属于这个proposal对应的cls
    #
    #     clu_point = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')  # M, num_point
    #     for i in range(num_instances):
    #         clu_point[
    #             proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1  # M , num_point 表示有哪些点被group到这个proposal
    #     final_proposals = clu_point * mask_pred  # M ,num_point 最终结果，最终的proposal中有哪些点
    #
    #     final_proposals = final_proposals.cpu().numpy()
    #     pred_bbox = np.zeros((num_instances, 6))  # M, 6 存储每个proposal的bbox的center和size
    #     for i in range(num_instances):
    #         idx = (final_proposals[i] == 1)
    #         object_points = batch['coords_float'][idx].cpu().numpy()
    #         if object_points.shape[0] != 0:
    #             max_corner = object_points.max(0)
    #             min_corner = object_points.min(0)
    #         else:
    #             max_corner = np.zeros(3)
    #             min_corner = np.zeros(3)
    #         center = (max_corner + min_corner) / 2
    #         size = abs(max_corner - min_corner)
    #         pred_bbox[i, 0:3] = center
    #         pred_bbox[i, 3:6] = size
    #
    #     pred_box_corner = get_3d_box_batch(pred_bbox[:, 3:6], np.zeros(num_instances), pred_bbox[:, 0:3])  # M, 8, 3
    #     pred_box_corner = torch.from_numpy(pred_box_corner).unsqueeze(0).cuda()  # 1, M, 8, 3
    #
    #     _, batch['object_assignment'], _, _ = nn_distance(torch.from_numpy(pred_bbox[:, 0:3]).unsqueeze(0).cuda(),
    #                                                       batch['center_label'])  # 1, M
    #
    #     # pick out object ids of detected objects
    #     detected_object_ids = torch.gather(batch["scene_object_ids"], 1, batch["object_assignment"])  # 1, M
    #
    #     # bbox corners
    #     assigned_target_bbox_corners = torch.gather(
    #         batch["gt_box_corner_label"],
    #         1,
    #         batch["object_assignment"].view(1, num_instances, 1, 1).repeat(1, 1, 8, 3)
    #     )  # 1, M, 8, 3
    #
    #     detected_bbox_corners = pred_box_corner  # 1, M, 8, 3
    #     # compute IoU between each detected box and each ground truth box
    #     ious = box3d_iou_batch_tensor(
    #         assigned_target_bbox_corners.view(-1, 8, 3),  # 1*M, 8, 3
    #         detected_bbox_corners.view(-1, 8, 3)  # 1*M, 8, 3
    #     ).view(1, num_instances)  # 1, M
    #
    #     # find good boxes (IoU > threshold)
    #     good_bbox_masks = ious > 0.5  # 1, M
    #     valid_bbox_masks = final_cls != 18  # M
    #     for prop_id in range(num_instances):
    #         if good_bbox_masks[0, prop_id] == 1:
    #             scene_id = str(batch['scan_ids'][0])
    #             object_id = str(detected_object_ids[0, prop_id].item())
    #             caption_decoded = decode_caption(captions[prop_id], self.vocabulary["idx2word"])
    #
    #             try:
    #                 ann_list = list(self.organized[scene_id][object_id].keys())
    #                 object_name = self.organized[scene_id][object_id][ann_list[0]]["object_name"]
    #                 # store
    #                 key = "{}|{}|{}".format(scene_id, object_id, object_name)
    #                 self.candidates[key] = [caption_decoded]
    #
    #             except KeyError:
    #                 continue
    #
    #     good_bbox_masks = good_bbox_masks.flatten()
    #     final_proposals = final_proposals[good_bbox_masks.cpu().numpy()]
    #     detected_object_ids = detected_object_ids.flatten()[good_bbox_masks.cpu().numpy()]
    #     # detected_object_ids = detected_object_ids.flatten()
    #     visual(final_proposals, batch['original_point'], detected_object_ids, scan_id)
    #
    #     # # 去除分数不高以及被分到background的proposal
    #     # inds = (max_cls_score > 0.2) * (final_cls != 18)
    #     # final_proposals = final_proposals[inds]
    #     # final_feats = feats[inds]
    #     #
    #     # # 去除点特别少的proposal
    #     # npoint = final_proposals.sum(1)
    #     # inds = npoint >= 100
    #     # final_proposals = final_proposals[inds]
    #     # final_feats = final_feats[inds]
    #     #
    #     # batch['final_feats'] = final_feats
    #     # batch['final_proposals'] = final_proposals.cpu().numpy()
    #     #
    #     # batch = self.caption_module.forward_scene_test(batch)
    #     # captions = batch['final_lang'].argmax(-1)
    #     # final_result = []
    #     # for i in range(captions.shape[0]):  # num_proposal
    #     #     dict = {}
    #     #     caption_decoded = decode_caption(captions[i], self.vocabulary["idx2word"])
    #     #     dict['object_id'] = i
    #     #     dict['point'] = batch['final_proposals'][i]
    #     #     dict['caption'] = caption_decoded
    #     #     final_result.append(dict)
    #     #
    #     # visual(final_result, batch['original_point'])
    #     # return None
    #
    # def on_test_end(self):
    #     corpus_path = os.path.join(CONF.PATH.OUTPUT, "corpus_val.json")
    #     if not os.path.exists(corpus_path):
    #         print("preparing corpus...")
    #         raw_data = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    #         corpus = prepare_corpus(raw_data, CONF.TRAIN.MAX_DES_LEN)
    #         with open(corpus_path, "w") as f:
    #             json.dump(corpus, f, indent=4)
    #     else:
    #         print("loading corpus...")
    #         with open(corpus_path) as f:
    #             corpus = json.load(f)
    #
    #     self.candidates = check_candidates(corpus, self.candidates)
    #     self.candidates = organize_candidates(corpus, self.candidates)
    #
    #     pred_path = os.path.join(CONF.PATH.OUTPUT, "pred_val.json")
    #     print("generating descriptions...")
    #     with open(pred_path, "w") as f:
    #         json.dump(self.candidates, f, indent=4)
    #
    #     print("computing scores...")
    #     bleu = capblue.Bleu(4).compute_score(corpus, self.candidates)
    #     cider = capcider.Cider().compute_score(corpus, self.candidates)
    #     rouge = caprouge.Rouge().compute_score(corpus, self.candidates)
    #     meteor = capmeteor.Meteor().compute_score(corpus, self.candidates)
    #
    #     print('BLEU-1,2,3,4 are:', bleu[0])
    #     print('CIDEr is:', cider[0])
    #     print('BLEU-4 is:', bleu[0][3])
    #     print('METEOR is:', meteor[0])
    #     print('ROUGE is:', rouge[0])
    #
    #     return None


def visual(final_proposals, original_point, detected_object_ids, scan_id):  # scan2cap论文可视化的是427
    # 测试，只取第一个
    for i in range(len(final_proposals)):
        idx = np.where(final_proposals[i] == 1)
        object_points = original_point[idx[0]]
        if not os.path.exists(os.path.join(CONF.PATH.OUTPUT, scan_id)):
            print('Creating new data folder: {}'.format(scan_id))
            os.mkdir(os.path.join(CONF.PATH.OUTPUT, scan_id))
        if idx[0].shape[0] != 0:
            box_min = np.min(object_points, axis=0)
            box_max = np.max(object_points, axis=0)
            color = np.asarray([220, 220, 60])
            output_path = os.path.join(CONF.PATH.OUTPUT, scan_id, 'object_id{}_bbox.ply'.format(detected_object_ids[i]))
            write_bbox(box_min, box_max, color, output_path)

    return None
