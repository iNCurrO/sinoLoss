import numpy as np
import torch
from typing import Tuple


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
            config,
            loss_list: Tuple[str] = tuple(["MSE"]),
            loss_weight: Tuple[float] = tuple([1.]),
    ):
        self._device = device
        self._network = network
        self._config = config
        assert self.is_available_losslist(loss_list), \
            "Implemention error: Not proper loss: {}(type:{}), where must be in {}".format(loss_list, type(loss_list), list(_loss_dict.keys()))
        self._loss_list = loss_list
        if 1-sum(loss_weight) < 1e-4:
            self._loss_weight = loss_weight
        else:
            self._loss_weight = [loss_weight_element/sum(loss_weight) for loss_weight_element in loss_weight]

    @staticmethod
    def is_available_losslist(loss_list):
        return all(map(lambda x: x in list(_loss_dict.keys()), loss_list))

    def run_denoiser(self, input_img):
        return self._network(input_img)

    def accumulate_gradients(self, input_img, target_img, Amatrix=None, targetsino=None):
        logs = ""
        for idx, loss in enumerate(self._loss_list):
            with torch.autograd.profiler.record_function(loss+"_forward"):
                denoised_img = self._network(input_img.requires_grad_(True))
                temp_loss = _loss_dict[loss](
                    denoised_img=denoised_img,
                    target_img=target_img,
                    Amatrix=Amatrix,
                    targetsino=targetsino,
                )
            with torch.autograd.profiler.record_function(loss+"_backward"):
                temp_loss.mul(self._loss_weight[idx]).backward()
            logs += f'{idx}. Loss of {loss}: {temp_loss}'
        return logs


@implemented_loss_list
def MSE(denoised_img, target_img, Amatrix=None, targetsino=None):
    MSEloss = torch.nn.MSELoss()
    return MSEloss(denoised_img, target_img)


@implemented_loss_list
def MAE(denoised_img, target_img, Amatrix=None, targetsino=None):
    MAEloss = torch.nn.L1Loss()
    return MAEloss(denoised_img, target_img)


@implemented_loss_list
def sinoloss_MSE(denoised_img, target_img, Amatrix=None, targetsino=None):
    denoised_sino = Amatrix(denoised_img)
    sino = Amatrix(target_img)
    MSEloss = torch.nn.MSELoss()
    return MSEloss(denoised_sino, sino)


@implemented_loss_list
def sinoloss_MAE(denoised_img, target_img, Amatrix=None, targetsino=None):
    denoised_sino = Amatrix(denoised_img)
    sino = Amatrix(target_img)
    MAEloss = torch.nn.L1Loss()
    return MAEloss(denoised_sino, sino)
