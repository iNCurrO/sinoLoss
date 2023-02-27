# import tigre # TODO Remove tigre option
import numpy as np
import torch


def forwardproejection(targetimg, Amatrix, config):
    batch_num = targetimg.shape[0]
    sinogram = torch.zeros(batch_num, 1, config.nDet, config.nView)
    for i in range(batch_num):
        tempimg = targetimg[i, :, :, :].permute(1, 2, 0)
        tempsino = forward_projector(tempimg, Amatrix, config)
        sinogram[i, :, :, :] = tempsino[None, :, :]
    return sinogram.cuda()


def forward_projector(targetimg, Amatrix, config):
    view = config.nView
    det_number = config.nDet
    sinogram = torch.zeros((det_number, view))
    for k in range(view):
        for n in range(det_number):
            temp_Amatrix = Amatrix[det_number*k + n]
            temp_weight = np.zeros(targetimg.shape)
            for coor, value in temp_Amatrix.items():
                xcor, ycor = divmod(coor, 512)
                temp_weight[xcor, ycor] = value
            sinogram[n, k] = torch.dot(
                torch.flatten(targetimg.cpu()),
                torch.flatten(torch.from_numpy(np.float32(temp_weight)))
            )
    return sinogram
