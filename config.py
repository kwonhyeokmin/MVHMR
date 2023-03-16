"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
import os
import os.path as osp
import argparse

class Config:
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    # root_dir = osp.join(cur_dir, '..')
    root_dir = cur_dir
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    result_dir = osp.join(output_dir, 'result')

    ## input, output
    CROP_IMG_HEIGHT = 256
    CROP_IMG_WIDTH = 192
    CROP_ASPECT_RATIO = CROP_IMG_HEIGHT / float(CROP_IMG_WIDTH)

    lr_dec_epoch = 45
    end_epoch = 60
    lr = 5e-5
    lr_dec_factor = 0.1
    batch_size = 16

    ## testing config
    use_gt_bbox = True
    num_thread = 8


cfg = Config()

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--version', help='Model name to eval', default='default', choices=['default', 'mvhmr'])
    parser.add_argument('--backbone', help='Backbone network of model', default='hr48', choices=['hr48', 'res50'])
    parser.add_argument('--datasets', help='Datasets for evaluation', default='h36m-p1', choices=['h36m-p1', 'h36m-p2', 'nia2023-p1', 'nia2023-p2'])
    parser.add_argument('--save-results', help='Save SMPL parameter that result of models', default=False)
    args = parser.parse_args()
    return args

from common.utils.dir_utils import add_pypath, make_folder

add_pypath(osp.join(cfg.root_dir, 'CLIFF'))

# Make folder
make_folder(cfg.output_dir)
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.result_dir)
