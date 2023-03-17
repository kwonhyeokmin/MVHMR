import os.path as osp
import numpy as np
from common.utils.pose_utils import world2cam, cam2pixel, process_bbox
import json
from common.utils import imutils
import constants
from glob import glob
import random
random.seed(2023)


class MultiViewNIA2023:
    def __init__(self, data_split, protocol):
        self.data_split = data_split
        self.img_dir = osp.join(constants.DATASET_FOLDERS[f'nia2023-p{protocol}'])
        self.annot_path = osp.join(constants.DATASET_FOLDERS[f'nia2023-p{protocol}'])
        self.joint_num = 29
        self.joints_name = (
            'pelvis', 'l_hip', 'l_knee', 'l_ankle', 'l_big_toe',
            'l_little_toe', 'r_hip', 'r_knee', 'r_ankle', 'r_big_toe',
            'r_little_toe', 'waist', 'chest', 'neck', 'l_shoulder',
            'l_elbow', 'l_wrist', 'l_mcp_joint_5', 'l_mcp_joint_2', 'r_shoulder',
            'r_elbow', 'r_wrist', 'r_mcp_joint_5', 'r_mcp_joint_2', 'nose',
            'l_eye', 'l_ear', 'r_eye', 'r_ear')
        self.lines = (
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 5),
            (0, 6), (6, 7), (7, 8), (8, 9), (9, 10), (8, 10),
            (0, 11), (11, 12), (12, 13),
            (13, 14), (14, 15), (15, 16), (16, 17), (16, 18), (17, 18),
            (13, 19), (19, 20), (20, 21), (21, 22), (21, 23), (22, 23),
            (13, 24), (24, 25), (25, 26), (24, 27), (27, 28)
        )
        self.root_idx = self.joints_name.index('pelvis')
        self.protocol = protocol
        self.joints_have_depth = True
        self.data, self.grouping = self.load_data()

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 0.9
        elif self.data_split == 'test':
            return 0.1
        else:
            assert 0, print('Unknown subset')

    def get_key_str(self, datum):
        return f"{osp.basename(datum['image']['path']).replace('.JPG', '').replace('_' + datum['camera']['position'], '')}"

    def get_ann(self, idx):
        ann = self.data[idx]
        return ann

    def load_data(self):
        print('Load data of NIA2023 Protocol ' + str(self.protocol))

        data = []
        grouping = {}
        data_id = 0
        json_file_lists = [x for x in glob(self.annot_path + f'/**/*.json', recursive=True)]
        anno_file_lists = []
        # skip if actor's height is lower than 150cm
        for file in json_file_lists:
            with open(file, 'r', encoding='utf-8') as fp:
                ann = json.load(fp)
            if ann['info']['actor']['height'] < 150.:
                continue
            anno_file_lists.append(file)
        if len(anno_file_lists) < 1:
            return data, grouping

        # split dataset
        ratio = self.get_subsampling_ratio()
        if self.data_split == 'train':
            anno_file_lists = anno_file_lists[:int((len(anno_file_lists) + 1) * ratio)]
        else:
            anno_file_lists = anno_file_lists[-int((len(anno_file_lists) + 1) * ratio):]

        for file in anno_file_lists:
            with open(file, 'r', encoding='utf-8') as fp:
                ann = json.load(fp)
            img_id = ann['info']['image']['id']
            img_path = ann['info']['image']['path']
            cam_idx = ['C', 'L', 'R'].index(ann['info']['camera']['position']) + 1
            f = np.array(ann['info']['camera']['focal_length'])
            c = np.array(ann['info']['camera']['principal_point'])

            extrinsic = np.array(ann['info']['camera']['extrinsic'])
            extrinsic_inv = np.linalg.inv(np.vstack((extrinsic, np.array([0, 0, 0, 1], dtype=np.float32))))
            extrinsic_inv[:3, 3] /= np.array(ann['info']['object'][0]['scale']) * 10
            extrinsic = np.linalg.inv(extrinsic_inv)[:3, :]
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]

            img_width = ann['info']['image']['width']
            img_height = ann['info']['image']['height']
            key_str = self.get_key_str(ann['info'])

            bbox = np.array(ann['annotation']['actor']['bbox']['2d'], dtype=np.int32)
            bbox = process_bbox(bbox, img_width, img_height)

            kp_world = np.array(ann['annotation']['actor']['keypoint']['3d'])
            kp_world /= np.array(ann['info']['object'][0]['scale']) * 10
            points_3d = world2cam(kp_world, R, t)
            points_2d = cam2pixel(points_3d, f, c)[:, :2]
            joint_vis = np.ones((self.joint_num, 1))

            if bbox is None: continue
            area = bbox[2] * bbox[3]

            bbox = np.array(imutils.xywh2xyxy(bbox))
            data.append({
                'id': data_id,
                'img_path': osp.join(self.img_dir, img_path),
                'img_id': img_id,
                'bbox': bbox,
                'area': area,
                'points_3d': points_3d,  # [org_img_x, org_img_y, depth]
                'points_2d': points_2d,
                'joint_vis': joint_vis,
                'focal': f[0],
                'princpt': c,
                'R': R,
                't': t,
                'key_str': key_str,
                'cam_idx': cam_idx,
            })

            if key_str not in grouping:
                grouping[key_str] = [-1, -1, -1]
            grouping[key_str][cam_idx - 1] = data_id
            data_id += 1

        return data, grouping
