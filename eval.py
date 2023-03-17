import os
from tqdm import tqdm

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset.dataset import MultiviewMocapDataset
from dataset.multiview_h36m import MultiViewH36M
from dataset.nia2023 import MultiViewNIA2023
from common.utils import dir_utils, cam_utils, visutils, renderer_pyrd, imutils
from common.utils.pose_utils import cam2world, cam2pixel, reconstruction_error
from common.utils import conversions
from config import cfg, make_args
import constants
from models.smpl import SMPL


if __name__ == '__main__':
    args = make_args()

    cudnn.benchmark = True
    backbone = args.backbone
    dataset_name = str(args.datasets)
    if args.version == 'default':
        eval("exec(f'from CLIFF.models.cliff_{backbone}.cliff import CLIFF')")
        print(f'Load cliff_{backbone} model')
        model = eval(f'CLIFF(constants.SMPL_MEAN_PARAMS)')
    elif args.version == 'mvhmr':
        eval("exec(f'from CLIFF.models.cliff_{backbone}.cliff_v2 import CLIFF_V2')")
        print(f'Load cliff_v2_{backbone} model')
        model = eval(f'CLIFF_V2(constants.SMPL_MEAN_PARAMS)')
    else:
        raise RuntimeError('Argument named version is invalid!')

    # Select device (gpu | cpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model checkpoint and make evaluation mode.
    model_ckpt = args.checkpoint
    state_dict = torch.load(model_ckpt, map_location=device)['model']
    state_dict = dir_utils.strip_prefix_if_present(state_dict, prefix="module.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    # Load SMPL model
    smpl = SMPL(constants.SMPL_MODEL_DIR,
                batch_size=cfg.batch_size,
                create_transl=True).to(device)

    # Load Datasets
    if 'h36m' in dataset_name:
        protocol = int(dataset_name.replace('h36m-p', ''))
        dataset = MultiViewH36M('test', protocol=protocol)
    elif 'nia2023' in dataset_name:
        protocol = int(dataset_name.replace('nia2023-p', ''))
        dataset = MultiViewNIA2023('test', protocol=protocol)
    else:
        AssertionError(f'{args.datasets} is not supported yet.')
    dataset_loader = MultiviewMocapDataset(dataset, True)
    print(f'The Number of Datasets: {len(dataset_loader)}')
    data_generator = DataLoader(dataset=dataset_loader, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

    # Regressor for Human3.6m
    np_J_regressor = np.load(constants.JOINT_REGRESSOR_H36M) if 'h36m' in dataset_name else np.load(constants.JOINT_REGRESSOR_MLKIT)
    J_regressor = torch.from_numpy(np_J_regressor).type(torch.FloatTensor).to(device)

    # Pose metrics
    # MPJPE and PA-MPJPE
    mpjpe = np.zeros(len(dataset_loader))
    pa_mpjpe = np.zeros(len(dataset_loader))

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset_loader), 72))
    smpl_betas = np.zeros((len(dataset_loader), 10))
    smpl_camera = np.zeros((len(dataset_loader), 3))
    pred_joints = np.zeros((len(dataset_loader), 17, 3))

    # joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_h36m = constants.H36M_TO_J14 if dataset_name in 'h36m' else [x for x in range(1, dataset.joint_num)]
    joint_mapper_smpl = constants.J24_TO_J14 if dataset_name in 'h36m' else [x for x in range(1, dataset.joint_num)]
    # joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # joint_mapper_gt = constants.J24_TO_J14
    vis = True

    for step, data in enumerate(tqdm(data_generator, desc='Eval', total=len(data_generator))):
        data_id = data['id']
        other_id = data['other_id']

        img_h = data['img_h'].to(device).float()
        img_w = data['img_w'].to(device).float()
        other_img_h = data['other_img_h'].to(device).float()
        other_img_w = data['other_img_w'].to(device).float()

        focal_length = data['focal'].to(device).float()
        princpt = data['princpt'].to(device).float()
        other_princpt = data['other_princpt'].to(device).float()
        points_3d = data['points_3d'].to(device).float()

        # target image
        norm_img = data['norm_img'].to(device).float()
        center = data['center'].to(device).float()
        scale = data['scale'].to(device).float()

        curr_batch_size = norm_img.shape[0]

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - princpt[:, 1], cy - princpt[:, 0], b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

        # other side image
        other_norm_img = data['other_norm_img'].to(device).float()
        other_center = data['other_center'].to(device).float()
        other_scale = data['other_scale'].to(device).float()
        other_focal_length = data['other_focal'].to(device).float()

        other_cx, other_cy, other_b = other_center[:, 0], other_center[:, 1], other_scale * 200
        other_bbox_info = torch.stack([other_cx - other_princpt[:, 1], other_cy - other_princpt[:, 0], other_b], dim=-1)

        other_bbox_info[:, :2] = other_bbox_info[:, :2] / other_focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        other_bbox_info[:, 2] = (other_bbox_info[:, 2] - 0.24 * other_focal_length) / (0.06 * other_focal_length)  # [-1, 1]

        with torch.no_grad():
            if args.version == 'default':
                pred_rotmat, pred_betas, pred_cam_crop = model(norm_img, bbox_info)
            else:
                pred_rotmat, pred_betas, pred_cam_crop, \
                    other_pred_rotmat, other_pred_betas, other_pred_cam_crop = model(
                    norm_img, bbox_info,
                    other_norm_img, other_bbox_info
                )

        pred_cam_full = cam_utils.cam_crop2full(pred_cam_crop, center, scale, princpt, focal_length)

        pred_output = smpl(betas=pred_betas,
                           body_pose=pred_rotmat[:, 1:],
                           global_orient=pred_rotmat[:, [0]],
                           pose2rot=False,
                           transl=pred_cam_full)
        pred_vertices = pred_output.vertices

        # Regressor broadcasting
        J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        gt_keypoints_3d = points_3d.to(device)

        if vis and (step + 1) % 100 == 1:
            for B in range(curr_batch_size):
                R = data['R'].to(device).float()[B]
                t = data['t'].to(device).float()[B]

                other_R = data['other_R'].to(device).float()[B]
                other_t = data['other_t'].to(device).float()[B]

                cam_pred_vertices = torch.from_numpy(cam2world(pred_vertices[B].detach().cpu().numpy(), R.detach().cpu().numpy(), t.detach().cpu().numpy())).float().to(device)

                target_img = imutils.load_img(dataset.get_ann(data_id[B])['img_path'])
                other_img = imutils.load_img(dataset.get_ann(other_id[B])['img_path'])
                ori_vis = np.hstack((target_img, cv2.resize(other_img, (target_img.shape[1], target_img.shape[0]))))

                # Visualization of mesh
                extrinsic = torch.eye(4, 4, device=device)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = t

                mesh_vis_img = renderer_pyrd.display(target_img,
                                                     cam_pred_vertices[:, :3].detach().cpu(),
                                                     focal=focal_length[B],
                                                     princpt=princpt[B],
                                                     extrinsics=extrinsic.detach().cpu(),
                                                     viewport_width=img_w[B],
                                                     viewport_height=img_h[B],
                                                     faces=smpl.faces)

                other_extrinsic = torch.eye(4, 4, device=device)
                other_extrinsic[:3, :3] = other_R
                other_extrinsic[:3, 3] = other_t

                mesh_other_vis_img = renderer_pyrd.display(other_img,
                                                           cam_pred_vertices[:, :3].detach().cpu(),
                                                           focal=other_focal_length[B],
                                                           princpt=other_princpt[B],
                                                           extrinsics=other_extrinsic.detach().cpu(),
                                                           viewport_width=other_img_w[B],
                                                           viewport_height=other_img_h[B],
                                                           faces=smpl.faces)
                mesh_vis = np.hstack((mesh_vis_img, cv2.resize(mesh_other_vis_img, (target_img.shape[1], target_img.shape[0]))))

                pred_points_2d = cam2pixel(pred_keypoints_3d[B].detach().cpu(),
                                           (focal_length.detach().cpu()[B], focal_length.detach().cpu()[B]),
                                           princpt.detach().cpu()[B])
                gt_points_2d = cam2pixel(gt_keypoints_3d[B].detach().cpu(),
                                         (focal_length.detach().cpu()[B], focal_length.detach().cpu()[B]),
                                         other_princpt.detach().cpu()[B])
                pred_joint_vis_img = visutils.vis_keypoints(target_img, pred_points_2d.transpose(), dataset.lines)
                gt_joint_vis_img = visutils.vis_keypoints(target_img, gt_points_2d.transpose(), dataset.lines)
                kp_vis = np.hstack((gt_joint_vis_img, cv2.resize(pred_joint_vis_img, (target_img.shape[1], target_img.shape[0]))))

                cv2.imwrite(os.path.join(cfg.vis_dir, f'{args.version}_{int(data_id[B])}_{B}_img.jpg'), np.vstack((np.vstack((ori_vis, kp_vis)), mesh_vis)))

        # Get 14 predicted joints from the mesh
        pred_pelvis = pred_keypoints_3d[:, [0], :]
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
        pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]

        gt_pelvis = gt_keypoints_3d[:, [0], :]
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
        gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_smpl, :]

        # MPJPE
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).detach().cpu().numpy()
        mpjpe[step * cfg.batch_size:step * cfg.batch_size + curr_batch_size] = error

        # PA-MPJPE
        pa_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
        pa_mpjpe[step * cfg.batch_size:step * cfg.batch_size + curr_batch_size] = pa_error

        if args.save_results:
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = conversions.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * cfg.batch_size:step * cfg.batch_size + curr_batch_size, :] = pred_pose.detach().cpu().numpy()
            smpl_betas[step * cfg.batch_size:step * cfg.batch_size + curr_batch_size, :] = pred_betas.detach().cpu().numpy()
            smpl_camera[step * cfg.batch_size:step * cfg.batch_size + curr_batch_size, :] = pred_cam_full.detach().cpu().numpy()

    if args.save_results:
        np.savez(os.path.join(cfg.result_dir, f'smpl_{backbone}-version_{args.version}_protocol_{dataset.protocol}'),
                 pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
        np.savez(os.path.join(cfg.result_dir, f'result_{backbone}-version_{args.version}_protocol_{dataset.protocol}'),
                 mpjpe=mpjpe, pa_mpjpe=pa_mpjpe)
    print('MPJPE: ' + str(1000 * mpjpe.mean()))
    print('PA-MPJPE: ' + str(1000 * pa_mpjpe.mean()))
