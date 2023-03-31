from config import get_config
from customlibs.chores import *
import torch
from forwardprojector.FP import FP
from model import unet
from customlibs.dataset import set_dataset
from customlibs.metrics import *
# Parse configuration
config = get_config()
model_init = {
    'UNET': lambda config, img_channel: unet.Unet(n_channels=img_channel),
}


def evaluate():
    # initialize dataset
    print(f"Data initialization: {config.dataname}\n")
    dataloader, valdataloader, num_channels = set_dataset(config)

    # Initiialize model
    print(f'Network initialization: {config.mode}\n')
    network = model_init[config.model.upper()](config, num_channels).cuda()

    # initialize optimzier
    optimizer = set_optimizer(config, network)

    # initialize Amatrix
    print(f"Amatrix initialization...")
    Amatrix = FP(config)
    print(f"Amatrix initialization finished!")

    # Set log
    print(f"Resume from: {config.resume}\n")
    __savedir__ = set_dir(config)
    print(f"New logs will be archived at the {__savedir__}\n")
    resume_network(config.resume, network, optimizer, config)
    network.eval()

    total_PSNR = 0.0
    total_SSIM = 0.0
    total_MSE = 0.0
    total_sinoMSE = 0.0
    cur_idx = 0
    final_idx = 0
    for batch_idx, samples in enumerate(valdataloader):
        [noisy_img, sino, _] = samples
        denoised_img = network(noisy_img.cuda())
        print(f'Calculate metrics... batchsize: {batch_idx-cur_idx}')
        total_SSIM += calculate_SSIM(denoised_img, noisy_img)/(batch_idx-cur_idx)
        total_PSNR += calculate_psnr(denoised_img, noisy_img)/(batch_idx-cur_idx)
        total_MSE += calculate_MSE(denoised_img, noisy_img)/(batch_idx-cur_idx)
        total_sinoMSE += calculate_sinoMSE(denoised_img, sino, Amatrix)/(batch_idx-cur_idx)
        final_idx = batch_idx
    log_str = f'Finished! SSIM: {total_SSIM}, PSNR: {total_PSNR}, MSE in image domain: {total_MSE}, ' \
              f'MSE in sino domain: {total_sinoMSE}\nFor total {final_idx}'
    print(log_str)
    with open(os.path.join(__savedir__, 'validation_logs.txt'), 'w') as log_file:
        print(log_str, file=log_file)


if __name__ == "__main__":
    evaluate()