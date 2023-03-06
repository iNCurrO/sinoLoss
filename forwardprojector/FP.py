import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import gc


# Using Parallel beam mode is not recommended: Not validated!
# Note that FP is extremely memory-consuming
# Therefore, to reduce the GPU memory consumption, most calculataion except forward is performed in CPU
# At least, 32GB RAM is recommended for 1024 view
# start in negative X-axis, counterclockwise
# ray-based parallelization is suported

class FP(nn.Module):
    def __init__(self, args):
        super(FP, self).__init__()

        self.args = args
        if self.args.datatype == 'float':
            self.args.datatype = torch.float
        elif self.args.datatype == 'double':
            self.args.datatype = torch.double

        # angle setting
        self.rot_dir = 1  # 1 for counterclockwise -1 for clockwise
        self.rot_deg = 360
        self.d_beta = np.pi * self.rot_deg / self.args.view / 180  # angular step size in radian
        self.beta = (self.rot_dir * torch.linspace(0, (self.args.view - 1) * self.d_beta, self.args.view,
                                                   dtype=self.args.datatype))
        self.args.DCD = self.args.SDD - self.args.SCD

        # Detector space, real, mm
        self.shift = 0
        if self.args.quarter_offset:
            self.shift = 0.25 * self.args.det_interval

        self.range_det = torch.tensor([-self.args.det_interval * (self.args.num_det - 1) / 2,
                                       self.args.det_interval * (self.args.num_det - 1) / 2],
                                      dtype=self.args.datatype) + self.shift

        self.param_setting()
        self.grid_pos, self.weighting = self.prepare(self.grid_pos, self.weighting)

    def param_setting(self):
        # Grid for Projection
        axis_x = (torch.linspace(-self.args.img_size[0] / 2, self.args.img_size[0] / 2, self.args.img_size[0] + 1,
                                 dtype=self.args.datatype) * self.args.pixel_size)  # cm scale, shape : (size+1) (1D)
        axis_y = (torch.linspace(-self.args.img_size[1] / 2, self.args.img_size[1] / 2, self.args.img_size[1] + 1,
                                 dtype=self.args.datatype) * self.args.pixel_size)

        if self.args.mode == 'equiangular':
            y_center = (self.args.num_det + 1) / 2 - self.shift / self.args.det_interval
            delta_gamma = self.args.det_interval / self.args.SDD
            gamma_vec = (torch.linspace(1, self.args.num_det, self.args.num_det,
                                        dtype=self.args.datatype) - y_center) * delta_gamma
            range_det_y = self.args.SDD * torch.sin(gamma_vec)
            range_det_x = (self.args.SDD * torch.cos(gamma_vec) - self.args.SCD)
        elif self.args.mode == 'equally_spaced':
            range_det_x = (torch.ones(self.args.num_det) * self.args.DCD)
            range_det_y = (torch.linspace(-(self.args.num_det - 1) / 2, (self.args.num_det - 1) / 2, self.args.num_det)
                           * self.args.det_interval + self.shift)

        src_zero = torch.tensor([-self.args.SCD, 0], dtype=self.args.datatype)
        a0 = torch.zeros(self.args.num_det, self.args.view, dtype=self.args.datatype)
        a1 = torch.ones(self.args.num_det, self.args.view, dtype=self.args.datatype)

        self.src_point = torch.cat(
            (src_zero[0] * torch.cos(self.beta)[None, :] + src_zero[1] * torch.sin(self.beta)[None, :],
             -src_zero[0] * torch.sin(self.beta)[None, :] + src_zero[1] * torch.cos(self.beta)[None, :]),
            dim=0)  # (2, view)

        self.det_x_rot = torch.cos(self.beta)[None, :] * range_det_x[:, None] + torch.sin(self.beta)[None, :] * \
                         range_det_y[:, None]  # (n_det, view)
        self.det_y_rot = -torch.sin(self.beta)[None, :] * range_det_x[:, None] + torch.cos(self.beta)[None, :] * \
                         range_det_y[:, None]  # (n_det, view)

        ax = (axis_x[:, None] - self.src_point[0][None, :])[None, :, :] / (self.det_x_rot - self.src_point[0])[:, None,
                                                                          :]  # (det, grid+1, view)
        ay = (axis_y[:, None] - self.src_point[1][None, :])[None, :, :] / (self.det_y_rot - self.src_point[1])[:, None,
                                                                          :]

        a_min = torch.maximum(torch.minimum(ax[:, 0, :], ax[:, -1, :]),
                              torch.minimum(ay[:, 0, :], ay[:, -1, :]))  # (det, view)
        a_min[a_min < 0] = 0
        a_max = torch.minimum(torch.maximum(ax[:, 0, :], ax[:, -1, :]),
                              torch.maximum(ay[:, 0, :], ay[:, -1, :]))  # (det, view)
        a_max[a_max > 1] = 1

        axy = torch.cat([ax, ay], dim=1)
        axy[torch.logical_or(axy > a_max[:, None, :], axy < a_min[:, None, :])] = float('nan')
        axy, _ = torch.sort(axy, dim=1)
        diff_axy = torch.diff(axy, n=1, dim=1)  # shape(det, x+y+1, view)
        diff_axy[torch.isnan(diff_axy)] = 0
        a_mid = axy[:, :-1, :] + diff_axy / 2  # shape(det, x+y+1, view)
        a_mid[torch.isnan(a_mid)] = 0
        s2d = torch.sqrt(torch.pow(self.det_x_rot - self.src_point[0], 2) + pow(self.det_y_rot - self.src_point[1],
                                                                                2))  # (det, view)
        self.weighting = diff_axy.permute(0, 2, 1)[None, None, :, :, :] * s2d[None, None, :, :,
                                                                          None]  # (batch, ch, det, view, x+y+1)

        x_pos = a_mid * (self.det_x_rot[:, None, :] - self.src_point[0, :]) + self.src_point[0, :]  # (det, x+y+1, view)
        x_pos = x_pos.permute(0, 2, 1) / (
                    self.args.pixel_size * (self.args.img_size[0] - 1) / 2)  # (det, view, x+y+1) normalize range
        y_pos = a_mid * (self.det_y_rot[:, None, :] - self.src_point[1, :]) + self.src_point[1, :]
        y_pos = y_pos.permute(0, 2, 1) / (self.args.pixel_size * (self.args.img_size[1] - 1) / 2)

        self.grid_pos = torch.cat((x_pos.unsqueeze(-1), y_pos.unsqueeze(-1)), dim=3)  # (det, view, x+y+1, 2)

    def forward(self, img):  # image is [batch, 1, X, Y] shape tensor
        self.batch, self.ch, _, _ = img.size()
        sinogram = self.prepare(
            torch.zeros([self.batch, self.ch, self.args.num_det, self.args.view], dtype=self.args.datatype))

        for i in range(self.args.view):
            interp_grid = F.grid_sample(img, torch.tile(self.grid_pos[:, i, :, :].view(1, self.args.num_det, -1, 2),
                                                        [self.batch, 1, 1, 1]),
                                        padding_mode="zeros", mode='bilinear', align_corners=False)
            interp_grid = interp_grid.view(self.batch, self.ch, self.args.num_det, -1)  # (batch, ch, det, x+y+1)
            sinogram[:, :, :, i] = torch.sum(self.weighting[:, :, :, i, :] * interp_grid, dim=-1)

        sinogram[torch.isnan(sinogram)] = 0
        return sinogram

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            return tensor.to(device)

        if len(args) <= 1:
            return args[0].to(device)
        else:
            return [_prepare(a) for a in args]
