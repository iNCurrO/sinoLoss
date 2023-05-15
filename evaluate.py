import os.path

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


def evaluate(network, valdataloader, Amatrix, saveimg=False, savedir = None):
    total_PSNR = 0.0
    total_SSIM = 0.0
    total_MSE = 0.0
    total_sinoMSE = 0.0
    num_data = len(valdataloader)
    for batch_idx, samples in enumerate(valdataloader):
        [noisy_img, sino, target_images] = samples
        denoised_img = network(noisy_img.cuda()).cpu()
        total_SSIM += calculate_SSIM(denoised_img, target_images)/num_data
        total_PSNR += calculate_psnr(denoised_img, target_images)/num_data
        total_MSE += calculate_MSE(denoised_img, target_images).detach().item()/num_data
        total_sinoMSE += calculate_sinoMSE(denoised_img.to(config.device), sino.to(config.device), Amatrix).detach().item()/num_data
        if saveimg:
            save_image
            save_images(
                target_images.cpu().detach().numpy(), 'target', str(batch_idx), os.path.join(savedir),
                config.valbatchsize
            )
            save_images(
                noisy_img.cpu().detach().numpy(), 'noisy', str(batch_idx), os.path.join(savedir),
                config.valbatchsize
            )
            save_images(
                denoised_img.cpu().detach().numpy(), 'denoised', str(batch_idx), os.path.join(savedir),
                config.valbatchsize
            )
        torch.cuda.empty_cache()
    return total_SSIM, total_PSNR, total_MSE, total_sinoMSE

def evaluate_main(resumenum=None, __savedir__=None):
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

    if not os.path.exists(os.path.join(__savedir__, 'test_result')):
        os.mkdir(os.path.join(__savedir__, 'test_result'))
    __savedir__ = os.path.join(__savedir__, 'test_result')

    print(f"Evaluation logs will be archived at the {__savedir__}\n")
    resume_network(resume=resumenum, network=network, optimizer=optimizer, config=config)
    network.eval()
    total_SSIM, total_PSNR, total_MSE, total_sinoMSE = evaluate(network, valdataloader, Amatrix, saveimg=True, savedir = __savedir__)

    log_str = f'Finished! SSIM: {total_SSIM}, PSNR: {total_PSNR}, '\
              f'MSE in image domain: {total_MSE}, ' \
              f'MSE in sino domain: {total_sinoMSE}\nFor total {len(valdataloader)}'
    print(log_str)
    with open(os.path.join(__savedir__, 'validation_logs.txt'), 'w') as log_file:
        print(log_str, file=log_file)


if __name__ == "__main__":
    temp_dir = [filename for filename in os.listdir(config.logdir) if filename.startswith(f"{int(config.resume.split('-')[0]):03}")]
    assert len(temp_dir) == 1, f'Duplicated file exists or non exist: {temp_dir}'
    evaluate_main(config.resume, os.path.join(config.logdir, temp_dir[0]))
