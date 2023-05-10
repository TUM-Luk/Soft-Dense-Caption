
import numpy as np
import os
import csv
from batch_load_scannet_data import export_one_scan

scan_name='scene0000_00'

data_vert = np.load("scannet_data/{}_vert.npy".format(scan_name))
print('vert.npy stores a numpy array with all point coordinates, it has shape', data_vert.shape)

data_aligned_vert = np.load("scannet_data/{}_aligned_vert.npy".format(scan_name))
print('\naligned_vert.npy stores a numpy array with all point coordinates after transformation, it has shape', data_aligned_vert.shape)

data_sem_label = np.load("scannet_data/{}_sem_label.npy".format(scan_name))
print('\nsem_label.npy stores a numpy array with label_id this point belong to, it has shape', data_sem_label.shape)

data_ins_label = np.load("scannet_data/{}_ins_label.npy".format(scan_name))
print('\nins_label.npy stores a numpy array with object_id this point belong to, it has shape', data_ins_label.shape)

data_bbox = np.load("scannet_data/{}_bbox.npy".format(scan_name))
print('\nbbox.npy stores a numpy array with object bounding box, it has shape', data_bbox.shape)
print("8 means: 6 for bounding box, 1 for label_id, 1 for object_id")

data_aligned_bbox = np.load("scannet_data/{}_aligned_bbox.npy".format(scan_name))
print('\nbbox.npy stores a numpy array with object bounding box after transformation, it has shape', data_aligned_bbox.shape)
print("8 means: 6 for bounding box, 1 for label_id, 1 for object_id")

point_votes = np.random.rand(10,3)
point_votes = np.tile(point_votes, (1, 3))
print(point_votes)



