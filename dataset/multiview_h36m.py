import os.path as osp
from pycocotools.coco import COCO
import numpy as np
from config import cfg
from common.utils.pose_utils import world2cam, cam2pixel, process_bbox
import json
from common.utils import imutils
import constants


class MultiViewH36M:
    def __init__(self, data_split, protocol):
        self.data_split = data_split
        self.img_dir = osp.join(constants.DATASET_FOLDERS['h36m'], 'images')
        self.annot_path = osp.join(constants.DATASET_FOLDERS['h36m'], 'annotations')
        self.human_bbox_dir = osp.join(constants.DATASET_FOLDERS['h36m'], 'bbox', 'bbox_human36m_output.json')
        self.joint_num = 17
        self.joints_name = (
            'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
            'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.lines = (
            (0,1),(1,2),(2,3),(0,4),(4,5),
            (5,6),(0,7),(7,8),(8,9),(9,10),
            (8,14),(14,15),(15,16),(8,11),(11,12),
            (12,13)
        )
        self.root_idx = self.joints_name.index('Pelvis')
        self.joints_have_depth = True
        self.protocol = protocol
        self.data, self.grouping = self.load_data()
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                            'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog',
                            'WalkTogether']

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            if self.protocol == 1:
                subject = [1, 5, 6, 7, 8, 9]
            elif self.protocol == 2:
                subject = [1, 5, 6, 7, 8]
        elif self.data_split == 'test':
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9, 11]
        else:
            assert 0, print("Unknown subset")

        return subject

    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_frame_{:06}'.format(
            datum['subject'], datum['action_idx'], datum['subaction_idx'],
            datum['frame_idx'])

    def get_ann(self, idx):
        ann = self.data[idx]
        return ann

    def load_data(self):
        print('Load data of H36M Protocol ' + str(self.protocol))
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()

        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'), 'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k, v in annot.items():
                    db.dataset[k] = v
            else:
                for k, v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                joints[str(subject)] = json.load(f)
        db.createIndex()

        if self.data_split == 'test' and not cfg.use_gt_bbox:
            print("Get bounding box from " + self.human_bbox_dir)
            bbox_result = {}
            with open(self.human_bbox_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_result[str(annot[i]['image_id'])] = np.array(annot[i]['bbox'])
        else:
            print("Get bounding box from groundtruth")

        data = []
        grouping = {}
        data_id = 0
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_width, img_height = img['width'], img['height']
            key_str = self.get_key_str(img)

            # check subject and frame_idx
            subject = img['subject']
            frame_idx = img['frame_idx']
            if subject not in subject_list:
                continue
            if frame_idx % sampling_ratio != 0:
                continue

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R, t, f, c = \
                np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(
                    cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)

            # project world coordinate to cam, image coordinate space
            action_idx = img['action_idx']
            subaction_idx = img['subaction_idx']
            frame_idx = img['frame_idx']
            kp_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)],
                                dtype=np.float32)
            points_3d = world2cam(kp_world, R, t) / 1000.
            points_2d = cam2pixel(points_3d, f, c)[:,:2]
            joint_vis = np.ones((self.joint_num, 1))
            # root_vis = np.array(ann['keypoints_vis'])[self.root_idx, None]

            # bbox load
            if self.data_split == 'test' and not cfg.use_gt_bbox:
                bbox = bbox_result[str(image_id)]
            else:
                bbox = np.array(ann['bbox'])

            bbox = process_bbox(bbox, img_width, img_height)

            if bbox is None: continue
            area = bbox[2] * bbox[3]

            bbox = np.array(imutils.xywh2xyxy(bbox))
            data.append({
                'id': data_id,
                'img_path': img_path,
                'img_id': image_id,
                'bbox': bbox,
                'area': area,
                'points_3d': points_3d,  # [org_img_x, org_img_y, depth]
                'points_2d': points_2d,
                'joint_vis': joint_vis,
                'focal': f[0],
                'princpt': c,
                'R': R,
                't': t * 0.001,
                'key_str': key_str,
                'cam_idx': cam_idx,
            })

            if key_str not in grouping:
                grouping[key_str] = [-1, -1, -1, -1]
            grouping[key_str][cam_idx - 1] = data_id
            data_id += 1

        return data, grouping
