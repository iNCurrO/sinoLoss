import pydicom
import torch
from forwardprojector.FP import FP
from config import get_config
import os
import matplotlib.pyplot as plt
import numpy as np
import forwardprojector.utility
import time

args = get_config()

device = torch.device('cpu' if args.cpu else 'cuda')
args.num_split = 1
args.view = 512
# args.ID_patient = ['L067','L096','L109','L143','L192']
args.ID_patient = ['L286','L291','L310','L333','L506']


def main():
    args.pixel_size = 0.7421875
    FP_model = FP(args)
    
    for i in args.patient_ID:
        path = os.path.join(args.data_dir, i, 'full_1mm')
        save_path = os.path.join('/Dataset', i, 'sinogram', str(args.view)+'views')
        print('save path is {}'.format(save_path))
        os.makedirs(save_path, exist_ok=True)
        filenames = os.listdir(path)
        for i in filenames:
            data = pydicom.dcmread(os.path.join(path, i))
            img = data.pixel_array*data.RescaleSlope
            img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
            tic = time.time()
            sinogram = FP_model(img.to(device))
            toc = time.time()
            sinogram = sinogram.squeeze().cpu().numpy()
            np.save(os.path.join(save_path, i[:-4]+'.npy'), sinogram)
            print('time: {}'.format(toc-tic))


if __name__ == '__main__':
    main()

