import numpy as np
import os.path as osp
import os
from operator import itemgetter

COLOR_DETECTRON2 = np.array([
    0.000, 0.447, 0.741,
    0.850, 0.325, 0.098,
    0.929, 0.694, 0.125,
    0.494, 0.184, 0.556,
    0.466, 0.674, 0.188,
    0.301, 0.745, 0.933,
    0.635, 0.078, 0.184,
    # 0.300, 0.300, 0.300,
    0.600, 0.600, 0.600,
    1.000, 0.000, 0.000,
    1.000, 0.500, 0.000,
    0.749, 0.749, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 1.000,
    0.667, 0.000, 1.000,
    0.333, 0.333, 0.000,
    0.333, 0.667, 0.000,
    0.333, 1.000, 0.000,
    0.667, 0.333, 0.000,
    0.667, 0.667, 0.000,
    0.667, 1.000, 0.000,
    1.000, 0.333, 0.000,
    1.000, 0.667, 0.000,
    1.000, 1.000, 0.000,
    0.000, 0.333, 0.500,
    0.000, 0.667, 0.500,
    0.000, 1.000, 0.500,
    0.333, 0.000, 0.500,
    0.333, 0.333, 0.500,
    0.333, 0.667, 0.500,
    0.333, 1.000, 0.500,
    0.667, 0.000, 0.500,
    0.667, 0.333, 0.500,
    0.667, 0.667, 0.500,
    0.667, 1.000, 0.500,
    1.000, 0.000, 0.500,
    1.000, 0.333, 0.500,
    1.000, 0.667, 0.500,
    1.000, 1.000, 0.500,
    0.000, 0.333, 1.000,
    0.000, 0.667, 1.000,
    0.000, 1.000, 1.000,
    0.333, 0.000, 1.000,
    0.333, 0.333, 1.000,
    0.333, 0.667, 1.000,
    0.333, 1.000, 1.000,
    0.667, 0.000, 1.000,
    0.667, 0.333, 1.000,
    0.667, 0.667, 1.000,
    0.667, 1.000, 1.000,
    1.000, 0.000, 1.000,
    1.000, 0.333, 1.000,
    1.000, 0.667, 1.000,
    # 0.333, 0.000, 0.000,
    0.500, 0.000, 0.000,
    0.667, 0.000, 0.000,
    0.833, 0.000, 0.000,
    1.000, 0.000, 0.000,
    0.000, 0.167, 0.000,
    # 0.000, 0.333, 0.000,
    0.000, 0.500, 0.000,
    0.000, 0.667, 0.000,
    0.000, 0.833, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 0.167,
    # 0.000, 0.000, 0.333,
    0.000, 0.000, 0.500,
    0.000, 0.000, 0.667,
    0.000, 0.000, 0.833,
    0.000, 0.000, 1.000,
    # 0.000, 0.000, 0.000,
    0.143, 0.143, 0.143,
    0.857, 0.857, 0.857,
    # 1.000, 1.000, 1.000
]).astype(np.float32).reshape(-1, 3) * 255
SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array([
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink',
    'bathtub', 'otherfurniture'
])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160]
}
SEMANTIC_IDX2NAME = {
    1: 'wall',
    2: 'floor',
    3: 'cabinet',
    4: 'bed',
    5: 'chair',
    6: 'sofa',
    7: 'table',
    8: 'door',
    9: 'window',
    10: 'bookshelf',
    11: 'picture',
    12: 'counter',
    14: 'desk',
    16: 'curtain',
    24: 'refridgerator',
    28: 'shower curtain',
    33: 'toilet',
    34: 'sink',
    36: 'bathtub',
    39: 'otherfurniture'
}


# generate .ply
def write_ply(verts, colors, indices, output_file, task, bbox_xyz):
    if task != "bbox_pred":
        if colors is None:
            colors = np.zeros_like(verts)
        if indices is None:
            indices = []

        file = open(output_file, 'w')
        file.write('ply \n')
        file.write('format ascii 1.0\n')
        file.write('element vertex {:d}\n'.format(len(verts)))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')
        file.write('property uchar red\n')
        file.write('property uchar green\n')
        file.write('property uchar blue\n')
        file.write('element face {:d}\n'.format(len(indices)))
        file.write('property list uchar uint vertex_indices\n')
        file.write('end_header\n')
        for vert, color in zip(verts, colors):
            file.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2],
                                                                int(color[0] * 255),
                                                                int(color[1] * 255),
                                                                int(color[2] * 255)))
        for ind in indices:
            file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
        file.close()
    else:
        # prime_corner = [0, 2, 5, 7]
        related_corner1 = [0, 1]
        related_corner2 = [0, 3]
        related_corner3 = [0, 4]
        related_corner4 = [2, 3]
        related_corner5 = [2, 6]
        related_corner6 = [2, 1]
        related_corner7 = [7, 3]
        related_corner8 = [7, 4]
        related_corner9 = [7, 6]
        related_corner10 = [5, 4]
        related_corner11 = [5, 6]
        related_corner12 = [5, 1]

        file = open(output_file, 'w')
        file.write('ply \n')
        file.write('format ascii 1.0\n')
        file.write('element vertex {:d}\n'.format(len(bbox_xyz)))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')
        file.write('property uchar red\n')
        file.write('property uchar green\n')
        file.write('property uchar blue\n')
        file.write('element edge {:d}\n'.format(len(bbox_xyz) // 2 * 3))
        file.write('property int vertex1\n')
        file.write('property int vertex2\n')
        file.write('property uchar red\n')
        file.write('property uchar green\n')
        file.write('property uchar blue\n')
        file.write('end_header\n')

        for i, corner_location in enumerate(bbox_xyz):
            file.write(
                '{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(corner_location[0], corner_location[1], corner_location[2],
                                                         int(255), int(0), int(0)))

        for i in range(len(bbox_xyz) // 8):
            if i == 0:
                cur_related_corner1 = related_corner1
                cur_related_corner2 = related_corner2
                cur_related_corner3 = related_corner3
                cur_related_corner4 = related_corner4
                cur_related_corner5 = related_corner5
                cur_related_corner6 = related_corner6
                cur_related_corner7 = related_corner7
                cur_related_corner8 = related_corner8
                cur_related_corner9 = related_corner9
                cur_related_corner10 = related_corner10
                cur_related_corner11 = related_corner11
                cur_related_corner12 = related_corner12
            else:
                cur_related_corner1 = np.sum([cur_related_corner1, [8, 8]], axis=0).tolist()
                cur_related_corner2 = np.sum([cur_related_corner2, [8, 8]], axis=0).tolist()
                cur_related_corner3 = np.sum([cur_related_corner3, [8, 8]], axis=0).tolist()
                cur_related_corner4 = np.sum([cur_related_corner4, [8, 8]], axis=0).tolist()
                cur_related_corner5 = np.sum([cur_related_corner5, [8, 8]], axis=0).tolist()
                cur_related_corner6 = np.sum([cur_related_corner6, [8, 8]], axis=0).tolist()
                cur_related_corner7 = np.sum([cur_related_corner7, [8, 8]], axis=0).tolist()
                cur_related_corner8 = np.sum([cur_related_corner8, [8, 8]], axis=0).tolist()
                cur_related_corner9 = np.sum([cur_related_corner9, [8, 8]], axis=0).tolist()
                cur_related_corner10 = np.sum([cur_related_corner10, [8, 8]], axis=0).tolist()
                cur_related_corner11 = np.sum([cur_related_corner11, [8, 8]], axis=0).tolist()
                cur_related_corner12 = np.sum([cur_related_corner12, [8, 8]], axis=0).tolist()

            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner1[0], cur_related_corner1[1],
                                                           int(255), int(0), int(0)))
            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner2[0], cur_related_corner2[1],
                                                           int(255), int(0), int(0)))
            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner3[0], cur_related_corner3[1],
                                                           int(255), int(0), int(0)))
            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner4[0], cur_related_corner4[1],
                                                           int(255), int(0), int(0)))
            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner5[0], cur_related_corner5[1],
                                                           int(255), int(0), int(0)))
            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner6[0], cur_related_corner6[1],
                                                           int(255), int(0), int(0)))
            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner7[0], cur_related_corner7[1],
                                                           int(255), int(0), int(0)))
            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner8[0], cur_related_corner8[1],
                                                           int(255), int(0), int(0)))
            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner9[0], cur_related_corner9[1],
                                                           int(255), int(0), int(0)))
            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner10[0], cur_related_corner10[1],
                                                           int(255), int(0), int(0)))
            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner11[0], cur_related_corner11[1],
                                                           int(255), int(0), int(0)))
            file.write('{:d} {:d} {:d} {:d} {:d}\n'.format(cur_related_corner12[0], cur_related_corner12[1],
                                                           int(255), int(0), int(0)))
        file.close()


def get_coords_color(prediction_path, room_name, task):
    # visualization_get_coords_color
    coord_file = osp.join(prediction_path, 'coords', room_name + '.npy')
    color_file = osp.join(prediction_path, 'colors', room_name + '.npy')
    label_file = osp.join(prediction_path, 'semantic_label', room_name + '.npy')
    # inst_label_file = osp.join(prediction_path, 'gt_instance', room_name + '.txt')
    xyz = np.load(coord_file)
    rgb = np.load(color_file)
    label = np.load(label_file)
    bbox_xyz = None

    # inst_label = np.array(open(inst_label_file).read().splitlines(), dtype=int)
    # inst_label = inst_label % 1000 - 1
    rgb = (rgb + 1) * 127.5

    if task == "semantic_pred":
        # visualization_semantic_pred_task
        semantic_file = os.path.join(prediction_path, 'semantic_pred', room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb
    elif task == "offset_semantic_pred":
        semantic_file = os.path.join(prediction_path, 'semantic_pred', room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

        offset_file = os.path.join(prediction_path, 'offset_pred', room_name + '.npy')
        assert os.path.isfile(offset_file), 'No offset result - {}.'.format(offset_file)
        offset_coords = np.load(offset_file)
        xyz += offset_coords
    elif task == "bbox_pred":
        bbox_corners_all_file = osp.join(prediction_path, 'bbox_corners_all', room_name + '.npy')
        assert os.path.isfile(bbox_corners_all_file), 'No bbox result - {}.'.format(bbox_corners_all_file)
        bbox_xyz = np.load(bbox_corners_all_file)
    else:
        print("not finsh")

    sem_valid = (label != -100)
    xyz = xyz[sem_valid]
    rgb = rgb[sem_valid]

    return xyz, rgb, bbox_xyz


if __name__ == '__main__':
    # visualization_config
    prediction_path = '/home/luk/DenseCap/visualization'
    room_name = 'scene0011_00'
    task_list = ["semantic_pred", "bbox_pred"]

    for _, task in enumerate(task_list):
        ply_out = '/home/luk/DenseCap/'
        xyz, rgb, bbox_xyz = get_coords_color(prediction_path, room_name, task=task)

        points = xyz[:, :3]
        colors = rgb / 255

        ply_out = osp.join(ply_out + room_name + "_" + task + ".ply")
        write_ply(points, colors, None, ply_out, task, bbox_xyz)
