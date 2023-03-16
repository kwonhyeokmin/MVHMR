# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
CROP_IMG_HEIGHT = 256
CROP_IMG_WIDTH = 192

H36M_ROOT = 'D:\Datasets\Human36M'
NIA2023_1_ROOT = 'D:/Datasets/NIA2023/122-1.다중객체3차원표현데이터(실내)/06.품질검증/1.Dataset'
NIA2023_2_ROOT = 'D:/Datasets/NIA2023/122-2.다중객체3차원표현데이터(실외)/06.품질검증/1.Dataset'

LSP_ROOT = ''
LSP_ORIGINAL_ROOT = ''
LSPET_ROOT = ''
MPII_ROOT = ''
COCO_ROOT = ''
MPI_INF_3DHP_ROOT = ''
PW3D_ROOT = ''
UPI_S1H_ROOT = ''

DATASET_FOLDERS = {
    'h36m': H36M_ROOT,
    'h36m-p1': H36M_ROOT,
    'h36m-p2': H36M_ROOT,
    'nia2023-p1': NIA2023_1_ROOT,
    'nia2023-p2': NIA2023_2_ROOT,
    'lsp-orig': LSP_ORIGINAL_ROOT,
    'lsp': LSP_ROOT,
    'lspet': LSPET_ROOT,
    'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
    'mpii': MPII_ROOT,
    'coco': COCO_ROOT,
    '3dpw': PW3D_ROOT,
    'upi-s1h': UPI_S1H_ROOT
}

"""
We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""
JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    'OP Nose',
    'OP Neck',
    'OP RShoulder',
    'OP RElbow',
    'OP RWrist',
    'OP LShoulder',
    'OP LElbow',
    'OP LWrist',
    'OP MidHip',
    'OP RHip',
    'OP RKnee',
    'OP RAnkle',
    'OP LHip',
    'OP LKnee',
    'OP LAnkle',
    'OP REye',
    'OP LEye',
    'OP REar',
    'OP LEar',
    'OP LBigToe',
    'OP LSmallToe',
    'OP LHeel',
    'OP RBigToe',
    'OP RSmallToe',
    'OP RHeel',
    # 24 Ground Truth joints (superset of joints from different datasets)
    'Right Ankle',
    'Right Knee',
    'Right Hip',
    'Left Hip',
    'Left Knee',
    'Left Ankle',
    'Right Wrist',
    'Right Elbow',
    'Right Shoulder',
    'Left Shoulder',
    'Left Elbow',
    'Left Wrist',
    'Neck (LSP)',
    'Top of Head (LSP)',
    'Pelvis (MPII)',
    'Thorax (MPII)',
    'Spine (H36M)',
    'Jaw (H36M)',
    'Head (H36M)',
    'Nose',
    'Left Eye',
    'Right Eye',
    'Left Ear',
    'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
JOINT_REGRESSOR_ORI = 'data/J_regressor_ori.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
JOINT_REGRESSOR_MLKIT = 'data/J_regressor_mlkit.npy'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# Human3.6m Evaluation joints
H36M_TO_J14 = [H36M_TO_J17[i] for i in [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]]
J24_TO_J14 = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]
