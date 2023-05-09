import pydicom
import torch
from forwardprojector.FBP import FBP
from config import get_config
from PIL import Image
import os
import numpy as np
from customlibs.chores import save_image
import time
import glob
import tqdm
import torch.nn.functional as F

args = get_config()
device = torch.device(args.device)
args.num_split = 1
args.noise=1e6
# binning_size = (4, 1)
# args.num_det = int(args.num_det / binning_size[0])
# args.det_interval *= binning_size[0]
# args.recon_interval = args.recon_interval * binning_size[0]
# args.recon_size = [int(args.recon_size[0] / binning_size[0]), int(args.recon_size[1] / binning_size[0])]
args.no_mask = True


def main():
    FBP_model = FBP(args)

    targetsino_list = glob.glob(os.path.join(args.datadir, args.originDatasetName + '_sinogram_' + str(args.view) + 'views', "*"))
    save_path = os.path.join(args.datadir, args.originDatasetName + '_recon_' + str(args.view) + 'views')
    os.makedirs(os.path.join(save_path, 'reconimage'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'sino'), exist_ok=True)
    print(f'save path is {save_path}')
    for i in targetsino_list:
        if os.path.splitext(i)[1] == '.npy':
            print(f'Loading... {i}')
            total_sinogram_np = np.load(i)
            total_sinogram = torch.FloatTensor(total_sinogram_np).unsqueeze(0).permute(1, 0, 2, 3).to(torch.device(args.device))
            for j in tqdm.trange(total_sinogram.shape[0]):
                # sinogram = F.avg_pool2d(total_sinogram.to(device), kernel_size=binning_size, stride=binning_size)
                recon_img = FBP_model(total_sinogram[j, :, :, :].unsqueeze(0))
                recon_img = recon_img.squeeze().cpu().numpy()
                np.save(os.path.join(os.path.join(save_path, 'reconimage'), os.path.basename(i)[:-4]+str(j)+'.npy'), recon_img)
                np.save(os.path.join(os.path.join(save_path, 'sino'), os.path.basename(i)[:-4]+str(j)+'sino.npy'), total_sinogram_np[j, :, :])
                # save_image(total_sinogram_np[j, :, :], os.path.join(save_path, os.path.basename(i)[:-4]+str(j)+'sino.png'), sino=True)
                # save_image(recon_img, os.path.join(save_path, os.path.basename(i)[:-4]+str(j)+'.png'), sino=False)


if __name__ == '__main__':
    main()
    print("Jobs Done")
