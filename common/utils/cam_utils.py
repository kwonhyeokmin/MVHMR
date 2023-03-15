import torch


def cam_crop2full(crop_cam, center, scale, princpt, focal_length):
    """ Convert the camera parameters from the crop camera to the full camera

    Args:
        crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
        center: shape=(N, 2) bbox coordinates (c_x, c_y)
        scale: shape=(N, 1) square bbox resolution  (b / 200)
        full_img_shape: shape=(N, 2) original image height and width
        focal_length: focal_length: shape=(N,)

    Returns:
        torch.Tensor: Converted camera parameters for full camera

    """
    # img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    # w_2, h_2 = img_w / 2., img_h / 2.
    w_2, h_2 = princpt[:, 1], princpt[:, 0]

    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def cam_full2crop(full_cam, center, scale, full_img_shape, focal_length):
    """ Convert the camera parameters from the full camera to the crop camera

    Args:
        full_cam: shape=(N, 3) weak perspective camera in full img coordinates (s, tx, ty)
        center: shape=(N, 2) bbox coordinates (c_x, c_y)
        scale: shape=(N, 1) square bbox resolution  (b / 200)
        full_img_shape: shape=(N, 2) original image height and width
        focal_length: focal_length: shape=(N,)

    Returns:
        torch.Tensor: Converted camera parameters for cropped camera

    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.

    tx = full_cam[0, 0]
    ty = full_cam[0, 1]
    tz = full_cam[0, 2]
    bs = 2 * focal_length / tz

    cam_x = (bs - 1e-9) / b
    cam_y = tx - (2 * (cx - w_2) / bs)
    cam_z = ty - (2 * (cy - h_2) / bs)
    crop_cam = torch.stack([cam_x, cam_y, cam_z], dim=-1)
    return crop_cam
