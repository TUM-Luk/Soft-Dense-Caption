""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py

Load Scannet scenes with vertices and ground truth labels for semantic and instance segmentations
"""

# python imports
import math
import os, sys, argparse
import inspect
import json
import pdb
import numpy as np
import scannet_utils

def read_aggregation(filename):
    object_id_to_segs = {} #object_id和segments的对应字典
    label_to_segs = {} #label和segments的对应字典
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups']) # 该场景中共有几个object(instance)
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label'] #该object对应的semantic label （如window，table...)
            segs = data['segGroups'][i]['segments'] #该object对应的所有segments （还不是很懂）
            object_id_to_segs[object_id] = segs #即对每个object——id，存储了它对应的segments

            # 在lebel_to_segs字典中，key为label（如window，table...),value为该label对应的所有segments
            # 例：如果该场景有semantic label “window”，在场景中有3个object对应该label，每个object对应的segments为[1,11],[2,22],[3,33]
            # 则该label "window" 对应的segments为[1,11,2,22,3,33]
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices']) #点的总数
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            # seg_to_verts字典中，存储了segments和点下标的对应关系
            # 例：在segment文件中，第0至第4个点都被归为了segment 1，则在字典中，key 1就对应一个点下标的列表[0,1,2,3,4]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, output_file=None):
    """ points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """

    #label_map 表示不同map对应的nyu40id，例如wall对应id1，chair对应id5（其中很多东西对应到了id 40，可能是不需要考虑的）
    label_map = scannet_utils.read_label_mapping(label_map_file, label_from='raw_category', label_to='nyu40id')

    # mesh_vertices存储点信息，返回Nx3或Nx6或Nx9的np数组，按需要选择3个函数（点，点+rgb，点+rgb+normal）
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)
    # mesh_vertices = scannet_utils.read_mesh_vertices_rgb_normal(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    axis_align_matrix = None

    # check if there is axisAlignment data (only train scene has)
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]

    # coordinatae transformation with axix_align_matrix
    if axis_align_matrix != None:
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        pts = np.ones((mesh_vertices.shape[0], 4))
        pts[:,0:3] = mesh_vertices[:,0:3] # using homogeneous coordinates
        pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
        aligned_vertices = np.copy(mesh_vertices)
        aligned_vertices[:,0:3] = pts[:,0:3]
    else:
        print("No axis alignment matrix found")
        aligned_vertices = mesh_vertices # for test scene, no transformation on coordinates

    # Load semantic and instance labels
    if os.path.isfile(agg_file):
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)
        seg_to_verts, num_verts = read_segmentation(seg_file)
        # 由以上的字典，则可以得出每个点对应的object_id和label

        # 创建列表，长度等于点的总数，存储每个点对应的label_id，0则代表该点不对应任何label
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        object_id_to_label_id = {}
        for label, segs in label_to_segs.items():
            label_id = label_map[label] #从lobel_map中，找到该label对应的id，如label_map['chair']=5
            for seg in segs:
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id

        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        num_instances = len(np.unique(list(object_id_to_segs.keys())))
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][0] #object_id与label_id对应的字典
        
        instance_bboxes = np.zeros((num_instances,8)) # also include object id
        aligned_instance_bboxes = np.zeros((num_instances,8)) # also include object id
        for obj_id in object_id_to_segs:
            label_id = object_id_to_label_id[obj_id]

            # bboxes in the original meshes
            obj_pc = mesh_vertices[instance_ids==obj_id, 0:3]
            if len(obj_pc) == 0: continue
            # Compute axis aligned box
            # An axis aligned bounding box is parameterized by
            # (cx,cy,cz) and (dx,dy,dz) and label id
            # where (cx,cy,cz) is the center point of the box,
            # dx is the x-axis length of the box.
            xmin = np.min(obj_pc[:,0])
            ymin = np.min(obj_pc[:,1])
            zmin = np.min(obj_pc[:,2])
            xmax = np.max(obj_pc[:,0])
            ymax = np.max(obj_pc[:,1])
            zmax = np.max(obj_pc[:,2])
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin, label_id, obj_id-1]) # also include object id
            # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
            instance_bboxes[obj_id-1,:] = bbox 

            # bboxes in the aligned meshes
            obj_pc = aligned_vertices[instance_ids==obj_id, 0:3]
            if len(obj_pc) == 0: continue
            # Compute axis aligned box
            # An axis aligned bounding box is parameterized by
            # (cx,cy,cz) and (dx,dy,dz) and label id
            # where (cx,cy,cz) is the center point of the box,
            # dx is the x-axis length of the box.
            xmin = np.min(obj_pc[:,0])
            ymin = np.min(obj_pc[:,1])
            zmin = np.min(obj_pc[:,2])
            xmax = np.max(obj_pc[:,0])
            ymax = np.max(obj_pc[:,1])
            zmax = np.max(obj_pc[:,2])
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin, label_id, obj_id-1]) # also include object id
            # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
            aligned_instance_bboxes[obj_id-1,:] = bbox 
    else:
        # use zero as placeholders for the test scene
        print("use placeholders")
        num_verts = mesh_vertices.shape[0]
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        instance_bboxes = np.zeros((1, 8)) # also include object id
        aligned_instance_bboxes = np.zeros((1, 8)) # also include object id

    if output_file is not None:
        np.save(output_file+'_vert.npy', mesh_vertices)
        np.save(output_file+'_aligned_vert.npy', aligned_vertices)
        np.save(output_file+'_sem_label.npy', label_ids)
        np.save(output_file+'_ins_label.npy', instance_ids)
        np.save(output_file+'_bbox.npy', instance_bboxes)
        np.save(output_file+'_aligned_bbox.npy', aligned_instance_bboxes)

    return mesh_vertices, aligned_vertices, label_ids, instance_ids, instance_bboxes, aligned_instance_bboxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_path', required=True, help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument('--label_map_file', required=True, help='path to scannetv2-labels.combined.tsv')
    opt = parser.parse_args()

    scan_name = os.path.split(opt.scan_path)[-1]
    mesh_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(opt.scan_path, scan_name + '.aggregation.json')
    seg_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(opt.scan_path, scan_name + '.txt') # includes axisAlignment info for the train set scans.
    export(mesh_file, agg_file, seg_file, meta_file, opt.label_map_file, opt.output_file)

if __name__ == '__main__':
    main()
