"""
using for debug
export datas of one scene 'scene0000_00'

Usage example: python ./test.py
"""

import numpy as np
import os
import csv
from batch_load_scannet_data import export_one_scan


SCANNET_DIR = 'scans'
SCAN_NAMES = sorted([line.rstrip() for line in open('meta_data/scannetv2.txt')])
LABEL_MAP_FILE = 'meta_data/scannetv2-labels.combined.tsv' # using nyu-id
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
MAX_NUM_POINT = 50000
OUTPUT_FOLDER = './scannet_data'

scan_name='scene0000_00'
output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)

if not os.path.exists(OUTPUT_FOLDER):
    print('Creating new data folder: {}'.format(OUTPUT_FOLDER))
    os.mkdir(OUTPUT_FOLDER)

export_one_scan(scan_name,output_filename_prefix)


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



