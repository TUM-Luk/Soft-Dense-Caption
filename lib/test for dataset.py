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

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.config import CONF
from lib.dataset import ScannetReferenceDataset
from utils.pc_utils import random_sampling, rotx, roty, rotz
from utils.box_util import get_3d_box, get_3d_box_batch
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# for model
from model.softgroup import SoftGroup
import torch

from dataset import get_scanrefer


DC = ScannetDatasetConfig()

file = '/home/luk/Downloads/softgroup_scannet_spconv2.pth'
net = torch.load(file)

train_model = SoftGroup().cuda()
train_model.load_state_dict(net['net'], strict=True)

print('AAAAAAAAAAAAAAAAAAA')

new_scanrefer_train, new_scanrefer_eval_train, new_scanrefer_eval_val, all_scene_list = get_scanrefer(model='')



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

dataloader_train = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

for batch in dataloader_train:
    print(train_model.forward(batch,return_loss=True))
    print(batch['semantic_labels'][0])
    break
