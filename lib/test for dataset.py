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
from model.proposal_module import ProposalModule
from model.caption_module import CaptionModule
import torch

from dataset import get_scanrefer

DC = ScannetDatasetConfig()

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader_train = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)

soft_model = SoftGroup().cuda()
file = '/home/luk/Downloads/epoch_10.pth'
net = torch.load(file)
soft_model.load_state_dict(net['net'], strict=True)

proposal_model = ProposalModule().cuda()
caption_model = CaptionModule(dataset.vocabulary,dataset.glove).cuda()

print(dataset.vocabulary['word2idx'])

for batch in dataloader_train:
    batch['clus_feats_batch'], batch['select_feats'], batch['losses'] = soft_model.forward(batch,return_loss=True)
    proposal_model.forward(batch)
    print(batch['pred_bboxes'])
    caption_model.forward(batch)
    print(batch['lang_cap'])
    print(batch['lang_cap'].shape)
    break
#
# for batch in dataloader_train:
#     print(batch['object_id'])
#     print(batch['object_id'].shape)
#     print(batch['lang_feat'])
#     print(batch['lang_feat'].shape)
#     print(batch['lang_len'])
#     print(batch['lang_len'].shape)
#     print(batch['lang_ids'])
#     print(batch['lang_ids'].shape)
#     break


# print(dataset[0]['instance_label'].max())
# print(dataset[0]['inst_num'])
# print(dataset[0]['instance_bboxes'][:,-1].max())