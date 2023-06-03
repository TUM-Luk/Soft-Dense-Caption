import json
import os
import torch
from lib.config import CONF
import numpy as np
import torch.nn.functional as F
import threading
from plyfile import PlyData, PlyElement

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

# caption_decoded= {'test': ['sos there is a black lujiachen eos']}
# gt_caption={'test': ['sos there is a black dog eos']}
#
# print(threading.Lock())
#
# bleu = capblue.Bleu(4).compute_score(gt_caption,caption_decoded)
# cider = capcider.Cider().compute_score(gt_caption,caption_decoded)
# rouge = caprouge.Rouge().compute_score(gt_caption,caption_decoded)
# meteor = capmeteor.Meteor().compute_score(gt_caption, caption_decoded)
# print(bleu)
# print(cider)
# print(rouge)
# print(meteor)


# print('test' in caption_decoded.keys())

a = np.zeros(10)
print(np.where(a == 1)[0].shape[0])

plydata = PlyData.read('/home/luk/DenseCap/data/scannet/scans/scene0427_00/scene0427_00_vh_clean_2.ply')
num_verts = plydata['vertex'].count

print(plydata['vertex'].data)

lines = open('/home/luk/DenseCap/data/scannet/scans/scene0427_00/scene0427_00.txt').readlines()
axis_align_matrix = None
for line in lines:
    if 'axisAlignment' in line:
        axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]

print(axis_align_matrix)
axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
vertices[:, 0] = plydata['vertex'].data['x']
vertices[:, 1] = plydata['vertex'].data['y']
vertices[:, 2] = plydata['vertex'].data['z']

pts = np.ones((vertices.shape[0], 4))
pts[:, 0:3] = vertices[:, 0:3]  # using homogeneous coordinates
pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
aligned_vertices = np.copy(vertices)
aligned_vertices[:, 0:3] = pts[:, 0:3]

plydata['vertex'].data['x'] = aligned_vertices[:, 0]
plydata['vertex'].data['y'] = aligned_vertices[:, 1]
plydata['vertex'].data['z'] = aligned_vertices[:, 2]


print(plydata['vertex'].data)

plydata.write('test0427.ply')

box_max = np.ones(3)
box_min = np.ones(3)*-1
color = np.random.rand(3) * 255
print(box_min)
print(box_max)
print(color)

