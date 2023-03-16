import numpy as np
import cv2
import copy
from torch.utils.data.dataset import Dataset
import random
from common.utils import imutils


class MultiviewMocapDataset(Dataset):
    def __init__(self, db, is_train):
        self.db = db.data
        self.grouping = db.grouping
        self.joint_num = db.joint_num
        self.root_idx = db.root_idx

        self.is_train = is_train

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False

    def load_rgb_img(self, path):
        img_bgr = imutils.load_img(path)
        if not isinstance(img_bgr, np.ndarray):
            raise IOError("Fail to read %s" % path)
        img_rgb = img_bgr[:, :, ::-1]
        img_h, img_w, channel = img_rgb.shape
        return img_rgb, (img_h, img_w)

    def __getitem__(self, index):
        data = copy.deepcopy(self.db[index])

        key_str = data['key_str']
        data['points_2d'] = np.array(data['points_2d'])
        data['points_3d'] = np.array(data['points_3d'])
        data['joint_vis'] = np.array(data['joint_vis'])
        cam_idx = data['cam_idx']

        # Load target image
        img_rgb, (img_h, img_w) = self.load_rgb_img(data['img_path'])

        norm_img, center, scale, crop_ul, crop_br, _ = \
            imutils.process_image(img_rgb, data['bbox'])
        # cv2.imwrite(f'{data["id"]}.jpg', _)
        data['norm_img'] = norm_img
        data['center'] = center
        data['scale'] = scale
        data['crop_ul'] = crop_ul
        data['crop_br'] = crop_br
        data['img_h'] = img_h
        data['img_w'] = img_w

        # Load other inputs
        group = copy.deepcopy(self.grouping[key_str])
        group.pop(cam_idx-1)
        # other_index = random.choice(group)
        other_index = group[0]
        other_data = copy.deepcopy(self.db[other_index])

        # Load other image
        other_img_rgb, (other_img_h, other_img_w) = self.load_rgb_img(other_data['img_path'])
        other_norm_img, other_center, other_scale, other_crop_ul, other_crop_br, _ = \
            imutils.process_image(other_img_rgb, other_data['bbox'])

        data['other_norm_img'] = other_norm_img
        data['other_center'] = other_center
        data['other_scale'] = other_scale
        data['other_crop_ul'] = other_crop_ul
        data['other_crop_br'] = other_crop_br
        data['other_img_h'] = other_img_h
        data['other_img_w'] = other_img_w

        data['other_id'] = other_data['id']
        data['other_focal'] = other_data['focal']
        data['other_princpt'] = other_data['princpt']
        data['other_R'] = other_data['R']
        data['other_t'] = other_data['t']

        data['other_points_2d'] = np.array(other_data['points_2d'])
        data['other_points_3d'] = np.array(other_data['points_3d'])
        data['other_joint_vis'] = np.array(other_data['joint_vis'])

        return data

    def __len__(self):
        return len(self.db)
