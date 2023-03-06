import pydicom
import torch
from forwardprojector.FBP import FBP
from config import get_config
import os
import matplotlib.pyplot as plt
import numpy as np
import forwardprojector.utility
import time
import shutil
import torch.nn.functional as F

args = get_config()
device = torch.device('cpu' if args.cpu else 'cuda')
args.num_split = 1
binning_size = (4, 1)
args.num_det = int(args.num_det / binning_size[0])
args.det_interval *= binning_size[0]
args.recon_interval = 0.7421875 * binning_size[0]
args.recon_size = [int(args.recon_size[0] / binning_size[0]), int(args.recon_size[1] / binning_size[0])]
args.no_mask = True
args.patient_ID = ['L067','L096','L109','L143','L192','L286','L291','L310','L333','L506']


def main():
    FBP_model = FBP(args)
    for patient_id in args.patient_ID:
        path = os.path.join('/Dataset', patient_id, 'sinogram', '512views')
        save_path = os.path.join('/Dataset', patient_id, 'recon_img', '512views', 'x'+str(binning_size[0]))
        os.makedirs(save_path, exist_ok=True)
        filenames = os.listdir(path)
        print('save path is {}'.format(save_path))
        for i in filenames:
            sinogram = np.load(os.path.join(path, i))
            sinogram = torch.FloatTensor(sinogram).unsqueeze(0).unsqueeze(0)
            sinogram = F.avg_pool2d(sinogram.to(device), kernel_size=binning_size, stride=binning_size)
            recon_img = FBP_model(sinogram)
            recon_img = recon_img.squeeze().cpu().numpy()
            np.save(os.path.join(save_path, i[:-4]+'.npy'), recon_img)


if __name__ == '__main__':
    main()
