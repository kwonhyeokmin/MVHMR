import torch
import numpy as np
from config import cfg
import copy
import constants


def cam2pixel(cam_coord, f, c, dtype='numpy'):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    if dtype == 'numpy':
        img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    elif dtype == 'torch':
        img_coord = torch.cat((x[:, None], y[:, None], z[:, None]), 1)
    else:
        AssertionError(f'{dtype} is not supported in this method')
    return img_coord


def pixel2cam(pixel_coord, f, c, dtype='numpy'):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    if dtype == 'numpy':
        cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    elif dtype == 'torch':
        cam_coord = torch.cat((x[:, None], y[:, None], z[:, None]), 1)
    else:
        AssertionError(f'{dtype} is not supported in this method')
    return cam_coord


def world2cam(world_coord, R, t, dtype='numpy'):
    if dtype == 'numpy':
        world_coord_kx4 = np.hstack((world_coord, np.ones((world_coord.shape[0], 1))))
        extrinsic = np.eye(3, 4)
        extrinsic[:3,:3] = R
        extrinsic[:3, 3] = t
        cam_coord = np.einsum('ij,kj->ki', extrinsic, world_coord_kx4)
    elif dtype == 'torch':
        device = world_coord.get_device()
        world_coord_kx4 = torch.hstack((world_coord, torch.ones((world_coord.shape[0], 1), device=device)))
        extrinsic = torch.eye(4, 4, device=device)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        cam_coord = torch.einsum('ij,kj->ki', extrinsic, world_coord_kx4)
    else:
        AssertionError(f'{dtype} is not supported in this method')
    return cam_coord[:,:3]


def cam2world(cam_coord, R, t, dtype='numpy'):
    if dtype == 'numpy':
        cam_coord_kx4 = np.hstack((cam_coord, np.ones((cam_coord.shape[0], 1))))
        extrinsic = np.eye(4, 4)
        extrinsic[:3,:3] = R
        extrinsic[:3, 3] = t
        world_coord = np.einsum('ij,kj->ki', np.linalg.inv(extrinsic), cam_coord_kx4)
    elif dtype == 'torch':
        device = cam_coord.get_device()
        cam_coord_kx4 = torch.hstack((cam_coord, torch.ones((cam_coord.shape[0], 1), device=device)))
        extrinsic = torch.eye(4, 4, device=device)
        extrinsic[:3,:3] = R
        extrinsic[:3, 3] = t
        world_coord = torch.einsum('ij,kj->ki', torch.linalg.inv(extrinsic), cam_coord_kx4)
    return world_coord[:,:3]


def get_bbox(joint_img):
    # bbox extract from keypoint coordinates
    bbox = np.zeros((4))
    xmin = np.min(joint_img[:, 0])
    ymin = np.min(joint_img[:, 1])
    xmax = np.max(joint_img[:, 0])
    ymax = np.max(joint_img[:, 1])
    width = xmax - xmin - 1
    height = ymax - ymin - 1

    bbox[0] = (xmin + xmax) / 2. - width / 2 * 1.2
    bbox[1] = (ymin + ymax) / 2. - height / 2 * 1.2
    bbox[2] = width * 1.2
    bbox[3] = height * 1.2

    return bbox


def process_bbox(bbox, width, height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = constants.CROP_IMG_HEIGHT / constants.CROP_IMG_WIDTH
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    # bbox[2] = w * 1.25
    # bbox[3] = h * 1.25
    bbox[2] = w
    bbox[3] = h
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.
    return bbox


def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re
