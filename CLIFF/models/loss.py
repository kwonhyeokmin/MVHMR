#  Copyright (c) 2022. Kwon., All rights reserved.
#  Kwon Hyeokmin <hykwon8952@gmail.com>

import torch
import torch.nn as nn


class Coord2DLoss(nn.Module):
    def __init__(self, weight=None):
        super(Coord2DLoss, self).__init__()

    def forward(self, coord_out, coord_gt):
        loss = torch.norm(coord_out - coord_gt, p=1, dim=-1, keepdim=True)
        return loss
