import torch
import torch.nn as nn
import numpy as np
import math
import os.path as osp

from common.utils.imutils import rot6d_to_rotmat, rotmat_to_rot6d
from models.backbones.hrnet.cls_hrnet import HighResolutionNet
from models.backbones.hrnet.hrnet_config import cfg
from models.backbones.hrnet.hrnet_config import update_config


class CLIFF_V2(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone"""

    def __init__(self, smpl_mean_params, img_feat_num=2048):
        super(CLIFF_V2, self).__init__()
        curr_dir = osp.dirname(osp.abspath(__file__))
        config_file = osp.join(curr_dir, "../backbones/hrnet/models/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
        update_config(cfg, config_file)
        self.encoder = HighResolutionNet(cfg)

        npose = 24 * 6
        nshape = 10
        ncam = 3
        nbbox = 3

        fc1_feat_num = 1024
        fc2_feat_num = 1024
        final_feat_num = fc2_feat_num
        reg_in_feat_num = img_feat_num + nbbox + npose + nshape + ncam
        # CUDA Error: an illegal memory access was encountered
        # the above error will occur, if use mobilenet v3 with BN, so don't use BN
        self.fc1 = nn.Linear(reg_in_feat_num, fc1_feat_num)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(fc1_feat_num, fc2_feat_num)
        self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(final_feat_num, npose)
        self.decshape = nn.Linear(final_feat_num, nshape)
        self.deccam = nn.Linear(final_feat_num, ncam)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def swap(self, R1, R2):
        batch_size = R1.shape[0]
        r1_9 = rot6d_to_rotmat(R1).view(batch_size, 24, 3, 3)
        r2_9 = rot6d_to_rotmat(R2).view(batch_size, 24, 3, 3)

        phi1 = r1_9[:,[0]]
        phi2 = r2_9[:,[0]]
        theta1 = r1_9[:,1:]
        theta2 = r2_9[:,1:]

        r1_9 = torch.cat((phi1, theta1), dim=1)
        r2_9 = torch.cat((phi2, theta2), dim=1)
        return rotmat_to_rot6d(r1_9).view(batch_size, -1), rotmat_to_rot6d(r2_9).view(batch_size, -1)

    def forward(self,
                x0, bbox0,
                x1, bbox1,
                init_pose0=None, init_shape0=None, init_cam0=None,
                init_pose1=None, init_shape1=None, init_cam1=None,
                n_iter=3):
        batch_size = x0.shape[0]
        if init_pose0 is None:
            init_pose0 = self.init_pose.expand(batch_size, -1)
        if init_shape0 is None:
            init_shape0 = self.init_shape.expand(batch_size, -1)
        if init_cam0 is None:
            init_cam0 = self.init_cam.expand(batch_size, -1)

        if init_pose1 is None:
            init_pose1 = self.init_pose.expand(batch_size, -1)
        if init_shape1 is None:
            init_shape1 = self.init_shape.expand(batch_size, -1)
        if init_cam1 is None:
            init_cam1 = self.init_cam.expand(batch_size, -1)

        xf0 = self.encoder(x0)
        xf1 = self.encoder(x1)

        pred_pose0 = init_pose0
        pred_shape0 = init_shape0
        pred_cam0 = init_cam0

        pred_pose1 = init_pose1
        pred_shape1 = init_shape1
        pred_cam1 = init_cam1

        for i in range(n_iter):
            # Swap pose output except pelvis
            phi0 = pred_pose0.clone()
            phi1 = pred_pose1.clone()
            pred_pose0 = torch.cat((pred_pose0[:, :6], phi1[:, 6:]), dim=1)
            pred_pose1 = torch.cat((pred_pose1[:, :6], phi0[:, 6:]), dim=1)
            pred_shape0, pred_shape1 = pred_shape1, pred_shape0

            xc0 = torch.cat([xf0, bbox0, pred_pose0, pred_shape0, pred_cam0], 1)
            xc0 = self.fc1(xc0)
            xc0 = self.drop1(xc0)
            xc0 = self.fc2(xc0)
            xc0 = self.drop2(xc0)
            pred_pose0 = self.decpose(xc0) + pred_pose0
            pred_shape0 = self.decshape(xc0) + pred_shape0
            pred_cam0 = self.deccam(xc0) + pred_cam0

            xc1 = torch.cat([xf1, bbox1, pred_pose1, pred_shape1, pred_cam1], 1)
            xc1 = self.fc1(xc1)
            xc1 = self.drop1(xc1)
            xc1 = self.fc2(xc1)
            xc1 = self.drop2(xc1)
            pred_pose1 = self.decpose(xc1) + pred_pose1
            pred_shape1 = self.decshape(xc1) + pred_shape1
            pred_cam1 = self.deccam(xc1) + pred_cam1

        pred_rotmat0 = rot6d_to_rotmat(pred_pose0).view(batch_size, 24, 3, 3)
        pred_rotmat1 = rot6d_to_rotmat(pred_pose1).view(batch_size, 24, 3, 3)

        return pred_rotmat0, pred_shape0, pred_cam0, pred_rotmat1, pred_shape1, pred_cam1
