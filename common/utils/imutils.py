#  Copyright (c) 2022. Kwon., All rights reserved.
#  Kwon Hyeokmin <hykwon8952@gmail.com>
#
#  This scripts are largely referred of CLIFF repository
#  (https://github.com/huawei-noah/noah-research/tree/master/CLIFF)

import torch
from torch.nn import functional as F
import numpy as np
import cv2

import constants
from config import cfg


def xywh2xyxy(xywh):
    """Change (x,y,width,height) format to (x1,y1,x2,y2) format"""
    x,y,w,h = xywh
    return [x, y, x+w, y+h]


def xyxy2xywh(xyxy):
    x1,y1,x2,y2 = xyxy
    return [x1, y1, x2-x1, y2-y1]


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    # res: (height, width), (rows, cols)
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        # To match direction of rotation from cropping
        rot = -rot
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1


def crop(img, center, scale, res):
    """ Crop image according to the supplied bounding box.
        res: [rows, cols]
    """
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    except Exception as e:
        print(e)

    new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

    return new_img, ul, br


def bbox_from_detector(bbox, rescale=1.1):
    """ Get center and scale of bounding box from bounding box.
        The expected format is [min_x, min_y, max_x, max_y].
    """
    # center
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center = torch.tensor([center_x, center_y])

    # scale
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = max(bbox_w * cfg.CROP_ASPECT_RATIO, bbox_h)
    scale = bbox_size / 200.0
    # adjust bounding box tightness
    scale *= (rescale / 1.2)
    return center, scale


def process_image(orig_img_rgb, bbox,
                  crop_height=constants.CROP_IMG_HEIGHT,
                  crop_width=constants.CROP_IMG_WIDTH):
    """ Read image, do preprocessing and possibly crop it according to the bounding box.
        If there are bounding box annotations, use them to crop the image.
        If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    try:
        center, scale = bbox_from_detector(bbox)
    except Exception as e:
        print("Error occurs in person detection", e)
        # Assume that the person is centered in the image
        height = orig_img_rgb.shape[0]
        width = orig_img_rgb.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width * crop_height / float(crop_width)) / 200.

    img, ul, br = crop(orig_img_rgb, center, scale, (crop_height, crop_width))
    crop_img = img.copy()

    img = img / 255.
    mean = np.array(constants.IMG_NORM_MEAN, dtype=np.float32)
    std = np.array(constants.IMG_NORM_STD, dtype=np.float32)
    norm_img = (img - mean) / std
    norm_img = np.transpose(norm_img, (2, 0, 1))

    return norm_img, center, scale, ul, br, crop_img


def rotmat_to_rot6d(x):
    """ Get 6D rotation representation from rotation matrix.

    Args:
        x (torch.Tensor): (B,3,3) Batch of corresponding rotation matrices

    Returns:
        torch.Tensor: (B,6) Batch of 6-D rotation representations

    References:
        Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019 (https://arxiv.org/abs/1812.07035)
    """
    x = x.view(-1, 3, 3)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    return torch.stack((a1, a2), dim=-1)


def rot6d_to_rotmat(x):
    """ Get rotation matrix from 6D rotation representation

    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations

    Returns:
        torch.Tensor: (B,3,3) Batch of corresponding rotation matrices

    References:
        Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019 (https://arxiv.org/abs/1812.07035)
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def save_obj(v, f, file_name='output.obj'):
    with open(file_name, 'w') as obj_file:
        for i in range(len(v)):
            obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
