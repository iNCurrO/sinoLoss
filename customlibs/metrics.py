import torch
import cv2
import numpy as np
from torch.nn.functional import mse_loss


def calculate_sinoMSE(img1, targetsino, Amatrix):
    inputsino = Amatrix(img1)
    return mse_loss(inputsino, targetsino)


def calculate_MSE(img1, img2):
    return mse_loss(img1, img2)


def calculate_psnr(img1, img2):
    mse = torch.mean((img1.cpu() - img2.cpu()) ** 2)
    assert torch.max(img1) < 10, f"Check the maximum value of img: {torch.max(img1)}"
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def calculate_SSIM(img1, img2):
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    img1 = torch.clamp(img1, min=0.0, max=1.0).detach().cpu().numpy()
    img2 = torch.clamp(img2, min=0.0, max=1.0).detach().cpu().numpy()
    img1 = img1.astype(np.float64).squeeze()
    img2 = img2.astype(np.float64).squeeze()
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


