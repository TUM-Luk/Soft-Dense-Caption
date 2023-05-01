import os
import sys
import time
import h5py
import json
import pickle
import random
import numpy as np
import multiprocessing as mp
from copy import deepcopy
from itertools import chain
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from lib.dataset import ScannetReferenceDataset
from utils.pc_utils import random_sampling, rotx, roty, rotz
from utils.box_util import get_3d_box, get_3d_box_batch
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis

DC=ScannetDatasetConfig()

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer(model=None):
    scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_train.json")))
    scanrefer_eval_train = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_train.json")))
    scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_val.json")))

    SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_train.json")))

    if model == 'debug':
        scanrefer_train = [SCANREFER_TRAIN[0]]
        scanrefer_eval_train = [SCANREFER_TRAIN[0]]
        scanrefer_eval_val = [SCANREFER_TRAIN[0]]

    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_eval_val])))

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    # eval on train
    new_scanrefer_eval_train = []
    for scene_id in train_scene_list:
        data = deepcopy(SCANREFER_TRAIN[0])
        data["scene_id"] = scene_id
        new_scanrefer_eval_train.append(data)

    new_scanrefer_eval_val = []
    for scene_id in val_scene_list:
        data = deepcopy(SCANREFER_TRAIN[0])
        data["scene_id"] = scene_id
        new_scanrefer_eval_val.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("using ScanRefer dataset")
    print("train on {} samples from {} scenes".format(len(new_scanrefer_train), len(train_scene_list)))
    print("eval on {} scenes from train and {} scenes from val".format(len(new_scanrefer_eval_train),
                                                                       len(new_scanrefer_eval_val)))
    return new_scanrefer_train, new_scanrefer_eval_train, new_scanrefer_eval_val, all_scene_list

SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")

new_scanrefer_train, new_scanrefer_eval_train, new_scanrefer_eval_val, all_scene_list = get_scanrefer(model='debug')
# print(new_scanrefer_train)


dataset = ScannetReferenceDataset(
        scanrefer=new_scanrefer_train,
        scanrefer_all_scene=all_scene_list,
        split='train',
        name="Scanrefer",
        num_points=40000,
        use_height=False,
        use_color=True,
        use_normal=False,
        use_multiview=False,
        augment=False,
        scan2cad_rotation=False
    )

# dataloader_train = DataLoader(dataset, batch_size=2, shuffle=True)

# for i in dataloader_train:
#     print(i)

# a=np.load('../data/scannet/scannet_data/scene0000_00_sem_label.npy')
# print(a[:40])
#

a = dataset[0]
print(a['semantic_label_nyu40id'])
print(a['semantic_label_cls'])
print(a["pt_offset_label"])
print(a["inst_num"])
print(a["num_bbox"])
print(a["inst_pointnum"])
print(a["inst_cls"])
print(a['sem_cls_label'])

