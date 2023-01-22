import numpy as np
import torch
from typing import Tuple
from forwardprojector import Ax, GenerateAmatrix


_loss_dict = dict()


def implemented_loss_list(fn: object) -> object:
    assert callable(fn)
    _loss_dict[fn.__name__] = fn
    return fn


class total_Loss:
    def __init__(
            self,
            device,
            network,
            loss_list: Tuple[str] = tuple(["MSE"]),
            loss_weight: Tuple[float] = tuple([1.]),
            CTGeo = None
    ):
        self._device = device
        self._network = network
        assert self.is_available_losslist(loss_list), \
            "Implemention error: Not proper loss: {}(type:{}), where must be in {}".format(loss_list, type(loss_list), list(_loss_dict.keys()))
        self._loss_list = loss_list
        if 1-sum(loss_weight) < 1e-4:
            self._loss_weight = loss_weight
        else:
            self._loss_weight = [loss_weight_element/sum(loss_weight) for loss_weight_element in loss_weight]
        if CTGeo is not None and "sino" in loss_list:
            print(f"Initializing Amatrix")
            self._Amatrix = GenerateAmatrix.GeoCal(CTGeo)
        else:
            self._Amatrix = None

    @staticmethod
    def is_available_losslist(loss_list):
        return all(map(lambda x: x in list(_loss_dict.keys()), loss_list))

    def run_denoiser(self, input_img):
        return self._network(input_img)

    def accumulate_gradients(self, input_img, target_img, input_sino=None):
        logs = []
        for idx, loss in enumerate(self._loss_list):
            with torch.autograd.profiler.record_function(loss+"_forward"):
                denoised_img = self._network(input_img.requires_grad_(True))
                temp_loss = _loss_dict[loss](denoised_img, target_img, input_sino, self._Amatrix)
            with torch.autograd.profiler.record_function(loss+"_backward"):
                temp_loss.mul(self._loss_weight[idx]).backward()
        return logs


@implemented_loss_list
def MSE(denoised_img, target_img, input_sino=None, Amatrix=None):
    MSEloss = torch.nn.MSELoss()
    return MSEloss(denoised_img, target_img)


@implemented_loss_list
def MAE(denoised_img, target_img, input_sino=None, Amatrix=None):
    MAEloss = torch.nn.L1Loss()
    return MAEloss(denoised_img, target_img)


@implemented_loss_list
def sinoloss_MSE(denoised_img, target_img, input_sino, Amatrix):
    denoised_sino = Ax.forwardproejection(denoised_img, Amatrix)
    MSEloss = torch.nn.MSELoss()
    return MSEloss(denoised_sino, input_sino)


@implemented_loss_list
def sinoloss_MAE(denoised_img, target_img, input_sino, Amatrix):
    denoised_sino = Ax.forwardproejection(denoised_img, Amatrix)
    MAEloss = torch.nn.L1Loss()
    return MAEloss(denoised_sino, input_sino)
