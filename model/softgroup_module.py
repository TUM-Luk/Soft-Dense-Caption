import functools
from collections import OrderedDict

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ops import (ball_query, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                 get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                 voxelization_idx)
from .util import cuda_cast, force_fp32, rle_decode, rle_encode
from .blocks import MLP, ResidualBlock, UBlock

from easydict import EasyDict

from lib.config import CONF


class SoftGroup(nn.Module):

    def __init__(self,
                 in_channels=6,
                 channels=32,
                 num_blocks=7,
                 semantic_classes=20,
                 instance_classes=18,
                 ignore_label=-100,
                 grouping_cfg=CONF.grouping_cfg,
                 instance_voxel_cfg=CONF.instance_voxel_cfg,
                 train_cfg=CONF.train_cfg,
                 test_cfg=CONF.test_cfg,
                 sem2ins_classes=[],
                 fixed_modules=[]):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.ignore_label = ignore_label
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules
        self.sem2ins_classes = sem2ins_classes

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # point-wise prediction
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)

        # topdown refinement path
        self.tiny_unet = UBlock([channels, 2 * channels], norm_fn, 2, block, indice_key_id=11)
        self.tiny_unet_outputlayer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # no-used
        self.cls_linear = nn.Linear(channels, instance_classes + 1)
        self.mask_linear = MLP(channels, instance_classes + 1, norm_fn=None, num_layers=2)
        self.iou_score_linear = nn.Linear(channels, instance_classes + 1)

        self.init_weights()

        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()
        for m in [self.cls_linear, self.iou_score_linear]:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def select_feat(self, proposals_idx, proposal_each_scene, instance_labels, object_id, batch_size):
        proposal_id_offset = np.zeros(batch_size + 1).astype(np.int64)
        good_clu_masks = torch.zeros(batch_size).bool().cuda()
        for i in range(batch_size + 1):
            proposal_id_offset[i] = sum(proposal_each_scene[0:i])
        # print('proposal_each_scene')
        # print(proposal_each_scene)
        # print('proposal_id_offset:')
        # print(proposal_id_offset)
        proposal_num = sum(proposal_each_scene)  # proposal总数量
        npoint = int(instance_labels.shape[0] / batch_size)  # 每个batch中点的数量，例如40000

        # 获取该batch中的ref_cluster (B,npoint) 如（4，40000）,为节约内存用bytetensor
        ref_cluster = torch.zeros((batch_size, npoint)).byte()
        for i in range(batch_size):
            ref_cluster[i, :] = (instance_labels[npoint * i:npoint * (i + 1)] == object_id[i])
        # print(ref_cluster)

        # 获取全部proposal的cluster (N,npoint) 如 (512,40000)
        proposal_cluster = torch.zeros((proposal_num, npoint)).byte()
        for [i, j] in proposals_idx:
            proposal_cluster[i, j % npoint] = 1
        # print(proposal_cluster)

        # 作iou选取和ref_cluster最接近的
        select_proposal_idx = torch.zeros(batch_size).int()
        for i in range(batch_size):
            ref = ref_cluster[i]
            iou_score = (proposal_cluster[proposal_id_offset[i]:proposal_id_offset[i + 1], :] * ref).sum(1) / (
                    proposal_cluster[proposal_id_offset[i]:proposal_id_offset[i + 1], :].sum(1) + ref.sum() - (
                    proposal_cluster[proposal_id_offset[i]:proposal_id_offset[i + 1], :] * ref).sum(1))
            if iou_score.shape[0] == 0:
                select_proposal_idx[i] = -1
            else:
                select_proposal_idx[i] = iou_score.argmax() + proposal_id_offset[i]
                if iou_score.max() > 0.2:
                    good_clu_masks[i] = 1

                # print(iou_score)
                # print(iou_score.argmax())
                # print("最匹配的cluster的下标为:", iou_score.argmax() + proposal_id_offset[i])

        return select_proposal_idx, proposal_id_offset, good_clu_masks

    @cuda_cast
    def forward(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                semantic_labels, instance_labels, pt_offset_labels, spatial_shape,
                batch_size, object_id, **kwargs):

        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(input, v2p_map)
        # print(output_feats.shape)
        # print(torch.unique(output_feats,dim=0).shape)
        proposals_idx, proposals_offset, proposal_each_scene = self.forward_grouping(semantic_scores, pt_offsets,
                                                                                     batch_idxs, coords_float,
                                                                                     self.grouping_cfg)

        select_proposal_idx, proposal_id_offset, good_clu_masks = self.select_feat(proposals_idx, proposal_each_scene,
                                                                                   instance_labels, object_id,
                                                                                   batch_size)

        # 提取每个instance proposal的feature
        # 这里的是voxelization之后的，即inst_feat是tiny unet的输入
        # inst_map是用来devoxelization的map
        inst_feats, inst_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats,
                                                          coords_float, rand_quantize=True, **self.instance_voxel_cfg)

        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)
        feats = self.global_pool(feats)

        # print(feats)
        # 对feats维度进行重构,从(N,32)变为(B,M,32)
        clus_feats_batch = torch.zeros([batch_size, self.train_cfg.max_proposal_num, feats.shape[1]]).cuda()
        for i in range(batch_size):
            clus_feats_batch[i][0:proposal_each_scene[i]] = feats[proposal_id_offset[i]:proposal_id_offset[i + 1]]

        select_feats = torch.zeros([batch_size, feats.shape[1]]).cuda()
        for i in range(batch_size):
            if select_proposal_idx[i] != -1:
                select_feats[i] = feats[select_proposal_idx[i]]

        # point wise losses
        losses = {}
        point_wise_loss = self.point_wise_loss(semantic_scores, pt_offsets, semantic_labels,
                                               instance_labels, pt_offset_labels)
        losses.update(point_wise_loss)

        print(good_clu_masks)
        return clus_feats_batch, select_feats, losses, good_clu_masks

    @cuda_cast
    def forward_val(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                    semantic_labels, instance_labels, pt_offset_labels, spatial_shape,
                    batch_size, object_id, **kwargs):

        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)

        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(input, v2p_map)

        proposals_idx, proposals_offset, proposal_each_scene = self.forward_grouping(semantic_scores, pt_offsets,
                                                                                     batch_idxs, coords_float,
                                                                                     self.grouping_cfg)

        proposal_id_offset = np.zeros(batch_size + 1).astype(np.int64)
        for i in range(batch_size + 1):
            proposal_id_offset[i] = sum(proposal_each_scene[0:i])

        num_proposal = sum(proposal_each_scene)
        clus_center = torch.zeros(num_proposal, 3).float().cuda()
        for i in range(num_proposal):
            coords_idx = proposals_idx[proposals_offset[i]:proposals_offset[i + 1]][1].long()
            center = torch.mean(coords_float[coords_idx], dim=0)
            clus_center[i] = center

        # 提取每个instance proposal的feature
        # 这里的是voxelization之后的，即inst_feat是tiny unet的输入
        # inst_map是用来devoxelization的map
        inst_feats, inst_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats,
                                                          coords_float, rand_quantize=True, **self.instance_voxel_cfg)

        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)
        feats = self.global_pool(feats)

        # print(feats)
        # 对feats维度进行重构,从(N,32)变为(B,M,32)
        clus_feats_batch = torch.zeros([batch_size, self.train_cfg.max_proposal_num, feats.shape[1]]).cuda()
        for i in range(batch_size):
            clus_feats_batch[i][0:proposal_each_scene[i]] = feats[proposal_id_offset[i]:proposal_id_offset[i + 1]]

        # 对centers进行重构，从(N,3)变为(B,M,3)
        clus_center_batch = torch.zeros([batch_size, self.train_cfg.max_proposal_num, 3]).cuda()
        for i in range(batch_size):
            clus_center_batch[i][0:proposal_each_scene[i]] = clus_center[
                                                             proposal_id_offset[i]:proposal_id_offset[i + 1]]

        # mask用来指示该位置是否是cluster，维度为(B,M)
        valid_clu_masks = torch.zeros(batch_size, self.train_cfg.max_proposal_num).bool().cuda()
        for i in range(batch_size):
            valid_clu_masks[i][0:proposal_each_scene[i]] = True

        # point wise losses
        losses = {}
        point_wise_loss = self.point_wise_loss(semantic_scores, pt_offsets, semantic_labels,
                                               instance_labels, pt_offset_labels)
        losses.update(point_wise_loss)

        return clus_feats_batch, clus_center_batch, valid_clu_masks, losses

    @cuda_cast
    def forward_visualization(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                              semantic_labels, instance_labels, pt_offset_labels, spatial_shape, batch_size,
                              scan_ids, **kwargs):
        color_feats = feats

        feats = torch.cat((feats, coords_float), 1)

        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        semantic_scores, pt_offsets, output_feats = self.forward_backbone(input, v2p_map)

        semantic_preds = semantic_scores.max(1)[1]
        ret = dict(scan_id=scan_ids[0])
        ret.update(
            dict(
                semantic_labels=semantic_labels.cpu().numpy(),
                instance_labels=instance_labels.cpu().numpy()))

        point_wise_results = self.get_point_wise_results(coords_float, color_feats,
                                                         semantic_preds, pt_offsets,
                                                         pt_offset_labels, v2p_map, lvl_fusion=False)
        ret.update(point_wise_results)

        proposals_idx, proposals_offset, proposal_each_scene = self.forward_grouping(
            semantic_scores,
            pt_offsets,
            batch_idxs,
            coords_float,
            self.grouping_cfg)

        inst_feats, inst_map = self.clusters_voxelization(proposals_idx, proposals_offset,
                                                          output_feats, coords_float,
                                                          **self.instance_voxel_cfg)

        _, cls_scores, iou_scores, mask_scores = self.forward_instance(inst_feats, inst_map)

        pred_instances = self.get_instances(
            scan_ids[0],
            proposals_idx,
            semantic_scores,
            cls_scores,
            iou_scores,
            mask_scores,
            v2p_map=v2p_map,
            lvl_fusion=False)

        gt_instances = self.get_gt_instances(semantic_labels, instance_labels)
        ret.update(dict(pred_instances=pred_instances, gt_instances=gt_instances))

        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)
        unselect_feats = self.global_pool(feats)
        ret.update(dict(unselect_feats=unselect_feats))

        return ret
    def point_wise_loss(self, semantic_scores, pt_offsets, semantic_labels, instance_labels,
                        pt_offset_labels):
        losses = {}
        semantic_loss = F.cross_entropy(
            semantic_scores, semantic_labels, weight=None, ignore_index=self.ignore_label)
        losses['semantic_loss'] = semantic_loss

        pos_inds = instance_labels != self.ignore_label
        if pos_inds.sum() == 0:
            offset_loss = 0 * pt_offsets.sum()
        else:
            offset_loss = F.l1_loss(
                pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / pos_inds.sum()
        losses['offset_loss'] = offset_loss
        return losses

    def forward_backbone(self, input, input_map, x4_split=False, lvl_fusion=False):
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features
        output_feats = output_feats[input_map.long()]  # devoxelization
        semantic_scores = self.semantic_linear(output_feats)
        pt_offsets = self.offset_linear(output_feats)
        return semantic_scores, pt_offsets, output_feats

    @force_fp32(apply_to=('semantic_scores, pt_offsets'))
    def forward_grouping(self,
                         semantic_scores,
                         pt_offsets,
                         batch_idxs,
                         coords_float,
                         grouping_cfg=None):
        proposal_each_scene = []
        proposals_idx_list = []
        proposals_offset_list = []
        batch_size = batch_idxs.max() + 1
        semantic_scores = semantic_scores.softmax(dim=-1)
        radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        npoint_thr = self.grouping_cfg.npoint_thr
        class_numpoint_mean = torch.tensor(
            self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)
        assert class_numpoint_mean.size(0) == self.semantic_classes

        npoint = (semantic_scores.shape[0] / batch_size).int()  # num of points in one sample/scene
        max_proposal = self.grouping_cfg.max_num_proposal  # TODO： max proposal per scene (sample)
        for i in range(batch_size):
            cur_num = 0  # 记录每个sample中当前group了多少proposal
            GO_NEXT = False  # 跳过当前sample的flag
            for class_id in range(self.semantic_classes):
                if GO_NEXT:
                    continue
                if class_id in self.grouping_cfg.ignore_classes:
                    continue
                scores = semantic_scores[npoint * i:npoint * (i + 1), class_id].contiguous()
                object_idxs = (scores > self.grouping_cfg.score_thr).nonzero().view(-1)
                if object_idxs.size(0) < self.test_cfg.min_npoint:
                    continue
                batch_idxs_ = batch_idxs[npoint * i + object_idxs]
                coords_ = coords_float[npoint * i + object_idxs]
                pt_offsets_ = pt_offsets[npoint * i + object_idxs]

                batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)  # 表明每个batch之间的分界，如[0,40000,80000]
                neighbor_inds, start_len = ball_query(
                    coords_ + pt_offsets_,
                    batch_idxs_,
                    batch_offsets_,
                    radius,
                    mean_active,
                    with_octree=False)
                proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, neighbor_inds.cpu(),
                                                              start_len.cpu(), npoint_thr, class_id)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int() + npoint * i

                # check if having max clusters, if yes, then crop, merge, and go to next sample
                cur_num = cur_num + (len(proposals_offset) - 1)
                if cur_num >= max_proposal:
                    crop_num = (len(proposals_offset) - 1) - (cur_num - max_proposal)  # 只能取前cur_num个proposal
                    proposals_offset = proposals_offset[0:crop_num + 1]
                    proposals_idx = proposals_idx[:proposals_offset[-1], :]
                    # set flag to go to next sample
                    cur_num = max_proposal
                    GO_NEXT = True

                # merge proposals
                if len(proposals_offset_list) > 0:
                    proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                    proposals_offset += proposals_offset_list[-1][-1]
                    proposals_offset = proposals_offset[1:]
                if proposals_idx.size(0) > 0:
                    proposals_idx_list.append(proposals_idx)
                    proposals_offset_list.append(proposals_offset)

            proposal_each_scene.append(cur_num)

        if len(proposals_idx_list) > 0:
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)
        else:
            proposals_idx = torch.zeros((0, 2), dtype=torch.int32)
            proposals_offset = torch.zeros((0,), dtype=torch.int32)
        return proposals_idx, proposals_offset, proposal_each_scene

    @force_fp32(apply_to='feats')
    def clusters_voxelization(self,
                              clusters_idx,
                              clusters_offset,
                              feats,
                              coords,
                              scale,
                              spatial_shape,
                              rand_quantize=False):
        if clusters_idx.size(0) == 0:
            # create dummpy tensors
            coords = torch.tensor(
                [[0, 0, 0, 0], [0, spatial_shape - 1, spatial_shape - 1, spatial_shape - 1]],
                dtype=torch.int,
                device='cuda')
            feats = feats[0:2]
            voxelization_feats = spconv.SparseConvTensor(feats, coords, [spatial_shape] * 3, 1)
            inp_map = feats.new_zeros((1,), dtype=torch.long)
            return voxelization_feats, inp_map

        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = sec_min(coords, clusters_offset.cuda())
        coords_max = sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = voxelization(feats, out_map.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats,
                                                     out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)
        return voxelization_feats, inp_map

    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    @force_fp32(apply_to=('x'))
    def global_pool(self, x, expand=False):
        indices = x.indices[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1,), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool

        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x

    def forward_instance(self, inst_feats, inst_map):
        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)

        # predict mask scores
        mask_scores = self.mask_linear(feats.features)
        mask_scores = mask_scores[inst_map.long()]
        instance_batch_idxs = feats.indices[:, 0][inst_map.long()]

        # predict instance cls and iou scores
        feats = self.global_pool(feats)
        cls_scores = self.cls_linear(feats)
        iou_scores = self.iou_score_linear(feats)
        return instance_batch_idxs, cls_scores, iou_scores, mask_scores

    @force_fp32(apply_to=('semantic_scores', 'cls_scores', 'iou_scores', 'mask_scores'))
    def get_instances(self,
                      scan_id,
                      proposals_idx,
                      semantic_scores,
                      cls_scores,
                      iou_scores,
                      mask_scores,
                      v2p_map=None,
                      lvl_fusion=False):
        if proposals_idx.size(0) == 0:
            return []

        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        semantic_pred = semantic_scores.max(1)[1]
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []
        for i in range(self.instance_classes):
            if i in self.sem2ins_classes:
                cls_pred = cls_scores.new_tensor([i + 1], dtype=torch.long)
                score_pred = cls_scores.new_tensor([1.], dtype=torch.float32)
                mask_pred = (semantic_pred == i)[None, :].int()
                if lvl_fusion:
                    mask_pred = mask_pred[:, v2p_map.long()]
            else:
                cls_pred = cls_scores.new_full((num_instances,), i + 1, dtype=torch.long)
                cur_cls_scores = cls_scores[:, i]
                cur_iou_scores = iou_scores[:, i]
                cur_mask_scores = mask_scores[:, i]
                score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
                mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')
                mask_inds = cur_mask_scores > self.test_cfg.mask_score_thr
                cur_proposals_idx = proposals_idx[mask_inds].long()
                mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1

                # filter low score instance
                inds = cur_cls_scores > self.test_cfg.cls_score_thr
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]

                if lvl_fusion:
                    mask_pred = mask_pred[:, v2p_map.long()]

                # filter too small instances
                npoint = mask_pred.sum(1)
                inds = npoint >= self.test_cfg.min_npoint
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]
            cls_pred_list.append(cls_pred.cpu())
            score_pred_list.append(score_pred.cpu())
            mask_pred_list.append(mask_pred.cpu())
        cls_pred = torch.cat(cls_pred_list).numpy()
        score_pred = torch.cat(score_pred_list).numpy()
        mask_pred = torch.cat(mask_pred_list).numpy()

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            instances.append(pred)
        return instances

    def get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = self.semantic_classes - self.instance_classes
        semantic_labels = semantic_labels - label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins.cpu().numpy()
        return gt_ins

    @force_fp32(apply_to=('semantic_preds', 'offset_preds'))
    def get_point_wise_results(self, coords_float, color_feats, semantic_preds, offset_preds,
                               offset_labels, v2p_map, lvl_fusion):
        if lvl_fusion:
            semantic_preds = semantic_preds[v2p_map.long()]
            offset_preds = offset_preds[v2p_map.long()]
        return dict(
            coords_float=coords_float.cpu().numpy(),
            color_feats=color_feats.cpu().numpy(),
            semantic_preds=semantic_preds.cpu().numpy(),
            offset_preds=offset_preds.cpu().numpy(),
            offset_labels=offset_labels.cpu().numpy())
